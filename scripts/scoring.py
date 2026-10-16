"""Task scoring from actual passage faces and physical contacts; never an actor."""
import numpy as np
from work.challenge2026.three_scenario_scene_v1.environment import EpisodeScorer


def declared_gap(case):
    gate=case.get('passage_gate')
    if gate is None:return None
    assert len(case['walls'])==2
    walls=sorted(case['walls'],key=lambda x:x['center_xyz_m'][0])
    assert all(abs(x['center_xyz_m'][1]-gate['y_m'])<1e-12 for x in walls)
    return [walls[0]['center_xyz_m'][0]+walls[0]['half_size_xyz_m'][0],
            walls[1]['center_xyz_m'][0]-walls[1]['half_size_xyz_m'][0]]


class PhysicalGapScorer(EpisodeScorer):
    def __init__(self,env):
        super().__init__(env);self.canonical_crossings=[]
        self.interval=declared_gap(env.case)
        if self.interval is not None:
            measured=[]
            for wall in env.case['walls']:
                gid=env.sim.model.geom_name2id('scenario_wall_'+wall['name']+'_geom')
                assert np.allclose(env.sim.data.geom_xmat[gid].reshape(3,3),np.eye(3),atol=1e-12,rtol=0)
                measured.append((float(env.sim.data.geom_xpos[gid,0]),float(env.sim.model.geom_size[gid,0])))
                assert abs(float(env.sim.data.geom_xpos[gid,1])-env.case['passage_gate']['y_m'])<1e-12
            measured.sort();actual=[measured[0][0]+measured[0][1],measured[1][0]-measured[1][1]]
            assert actual==self.interval

    def update_route(self,sample,command_index,phase='policy'):
        xyz=np.asarray(sample['plate_xyz'],float);gate=self.env.case.get('passage_gate')
        if gate and self.previous[1]<gate['y_m']<=xyz[1]:
            a=(gate['y_m']-self.previous[1])/(xyz[1]-self.previous[1]);cross=self.previous+a*(xyz-self.previous)
            if abs(cross[2]-self.env.initial_plate_xyz[2])<=gate['maximum_height_change_m']:
                event=dict(command_index=int(command_index),phase=phase,crossing_xyz_m=cross.tolist())
                if not self.crossings and self.interval[0]<=cross[0]<=self.interval[1]:self.crossings.append(event)
                lo,hi=gate['plate_centre_x_interval_m']
                if not self.canonical_crossings and lo<=cross[0]<=hi:self.canonical_crossings.append(event)
        self.previous=xyz

    def result(self,monitor):
        result=super().result(monitor)
        result.update(passage_scoring_rule='actual_inner_face_crossing_and_all_substep_contacts',
            canonical_declared_gate_route_complete=bool(self.canonical_crossings) if self.interval else True,
            canonical_declared_gate_crossings=self.canonical_crossings,
            actual_gap_interval_x_m=self.interval,geometry_used_only_for_reward_and_scoring=True)
        return result

