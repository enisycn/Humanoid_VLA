"""Flow matching plus full-field replay distillation, used in the reported fit."""
from pathlib import Path

def masked_field_mse(prediction, reference, action_is_pad):
    """Per-context reduction, then mean: equal context weighting as stock BC."""
    assert prediction.shape == reference.shape and prediction.ndim == 3
    assert action_is_pad.shape == prediction.shape[:2]
    mask = (~action_is_pad).unsqueeze(-1)
    denom = ((~action_is_pad).sum(dim=1) * prediction.shape[-1]).clamp_min(1)
    return (((prediction - reference).square() * mask).sum(dim=(1, 2)) / denom).mean()

class FrozenExpertReference:
    """Exact e22 expert values without constructing a second SmolVLA network.

    functional_call swaps only the named expert tensors for this call. Frozen
    VLM tensors stay shared. The teacher is eval/no_grad; explicit noise/time
    plus RNG equality checks prevent teacher computation changing BC randomness.
    """
    def __init__(self, student, names):
        import torch
        assert len(names) == 299
        self.student = student
        self.names = set(names)
        self.parameters = {k:v.detach().clone() for k,v in student.named_parameters() if k in names}
        assert len(self.parameters) == 299
        assert all(v.dtype == torch.float32 and not v.requires_grad for v in self.parameters.values())
        self.original_ids = {k:id(v) for k,v in student.named_parameters()}
        self.original_buffer_ids = {k:id(v) for k,v in student.named_buffers()}

    def velocity(self, batch, noise, times):
        import torch
        from torch.func import functional_call
        model = self.student
        flags = [(m, m.training) for m in model.modules()]
        cpu_rng = torch.get_rng_state().clone()
        device = noise.device
        device_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
        found = []
        hook = model.model.action_out_proj.register_forward_hook(lambda module, inputs, out: found.append(out.detach()))
        try:
            model.eval()
            with torch.no_grad():
                functional_call(model, self.parameters, (batch,),
                    dict(noise=noise, time=times, reduction='none'), strict=False)
        finally:
            hook.remove()
            # Assign individually: Module.train() would recursively overwrite
            # the VLM's deliberately frozen eval mode.
            for module, flag in flags:
                module.training = flag
        assert len(found) == 1 and found[0].shape == (24, 50, 32)
        assert not found[0].requires_grad and torch.isfinite(found[0]).all()
        assert torch.equal(cpu_rng, torch.get_rng_state()), 'Reference consumed CPU RNG'
        if device_rng is not None:
            assert torch.equal(device_rng, torch.cuda.get_rng_state(device)), 'Reference consumed CUDA RNG'
        assert self.original_ids == {k:id(v) for k,v in model.named_parameters()}
        assert self.original_buffer_ids == {k:id(v) for k,v in model.named_buffers()}
        return found[0]

    def verify_exact(self, parent):
        import torch
        from safetensors import safe_open
        with safe_open(str(Path(parent) / 'model.safetensors'), framework='pt', device='cpu') as s:
            for name, value in self.parameters.items():
                expected = s.get_tensor(name)
                assert value.dtype == expected.dtype == torch.float32
                assert torch.equal(value.cpu(), expected) and not value.requires_grad
        return True

def capture_student_velocity(student, batch, noise, times):
    found = []
    hook = student.model.action_out_proj.register_forward_hook(lambda module, inputs, out: found.append(out))
    try:
        losses, _ = student.forward(batch, noise=noise, time=times, reduction='none')
    finally:
        hook.remove()
    assert len(found) == 1 and found[0].shape == (24, 50, 32)
    return losses, found[0]

def forward_objective(student, reference, value, weight):
    import torch
    student.train()
    student.model.vlm_with_expert.vlm.eval()
    actions = student.prepare_action(value)
    assert actions.shape == (24, 50, 32)
    # Same order and draws as stock forward: noise first, then Beta time.
    noise = student.model.sample_noise(actions.shape, actions.device)
    times = student.model.sample_time(actions.shape[0], actions.device)
    target = reference.velocity(value, noise, times)
    losses, velocity = capture_student_velocity(student, value, noise, times)
    assert losses.shape == (24,) and torch.isfinite(losses).all()
    assert velocity.dtype == target.dtype == torch.float32 and torch.isfinite(velocity).all()
    kd = masked_field_mse(velocity[6:], target[6:], value['action_is_pad'][6:])
    bc = losses.mean()
    total = bc + weight * kd
    assert torch.isfinite(total)
    # Independent first7 arithmetic check against official stock BC reduction.
    valid = (~value['action_is_pad']).unsqueeze(-1)
    u = noise - actions
    manual = (((velocity[:, :, :7] - u[:, :, :7]).square() * valid).sum(dim=(1,2)) /
        ((~value['action_is_pad']).sum(dim=1) * 7).clamp_min(1))
    assert torch.allclose(manual, losses, rtol=1e-6, atol=1e-7)
    return total, bc, kd, losses
