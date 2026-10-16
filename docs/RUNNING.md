# Running the project

The recorded experiments use separate native-simulation, SmolVLA and DINOv2 environments. The repository is being packaged with the exact runtime and model identities; the existing results and videos can already be inspected without GPU setup.

```bash
git clone https://github.com/enisycn/Humanoid_VLA.git
cd Humanoid_VLA
python -m http.server 8000
```

Open README.md on GitHub, or download the MP4 files in `media/`. `results/metrics.csv` contains every completed control outcome. B means previous policy; J means final policy. Neither is the original public baseline. The JSON files retain original experiment paths for audit only; they are not portable run commands.

Do not infer broad pretrained-task retention from the three pushing controls. A fresh-machine reproduction has not yet been verified.
