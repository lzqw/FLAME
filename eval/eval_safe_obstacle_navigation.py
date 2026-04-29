import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse, json
from pathlib import Path
import numpy as np
from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv
from scripts.train_safe_obstacle_navigation import sample_policy


def classify_route(pos_traj):
    near_idx = np.where(np.abs(pos_traj[:, 0]) < 0.5)[0]
    y_mean = np.mean(pos_traj[near_idx, 1]) if len(near_idx) > 0 else np.mean(pos_traj[:, 1])
    return 'upper' if y_mean > 0 else 'lower'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--algo', required=True); p.add_argument('--eval_episodes', type=int, default=200); p.add_argument('--save_dir', required=True); args=p.parse_args()
    save_dir=Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    env = SafeObstacleNavigation2DEnv(use_filter=args.algo!='rf2_no_filter', seed=0)
    T=env.episode_len; N=args.eval_episodes
    positions=np.zeros((N,T+1,2),np.float32); obs_arr=np.zeros((N,T+1,8),np.float32); raw=np.zeros((N,T,2),np.float32); exe=np.zeros((N,T,2),np.float32); rewards=np.zeros((N,T),np.float32)
    sv=np.zeros((N,T),bool); tv=np.zeros((N,T),bool); safv=np.zeros((N,T),bool); filt=np.zeros((N,T),bool); pres=np.zeros((N,T),np.float32); pcost=np.zeros((N,T),np.float32); dgoal=np.zeros((N,T),np.float32); dobs=np.zeros((N,T),np.float32)
    succ=np.zeros((N,),bool); ttg=np.full((N,),T,np.int32); epret=np.zeros((N,),np.float32)
    routes=[]
    for i in range(N):
        obs,_=env.reset(seed=i); positions[i,0]=env.state; obs_arr[i,0]=obs
        for t in range(T):
            a=sample_policy(args.algo, env.state, env.goal)
            nxt,r,term,trunc,info=env.step(a)
            raw[i,t]=info['raw_action']; exe[i,t]=info['exec_action']; rewards[i,t]=r
            sv[i,t]=info['state_violation']; tv[i,t]=info['tightened_violation']; safv[i,t]=info['safe_violation']; filt[i,t]=info['filter_active']
            pres[i,t]=info['projection_residual']; pcost[i,t]=info['projection_cost']; dgoal[i,t]=info['distance_to_goal']; dobs[i,t]=info['distance_to_obstacle']
            positions[i,t+1]=env.state; obs_arr[i,t+1]=nxt; epret[i]+=r
            if term and not succ[i]: succ[i]=True; ttg[i]=t+1
            if term or trunc: break
        if succ[i]: routes.append(classify_route(positions[i,:ttg[i]+1]))
    np.savez(save_dir/'rollouts.npz', positions=positions, obs=obs_arr, raw_actions=raw, exec_actions=exe, rewards=rewards,
             state_violation=sv, tightened_violation=tv, safe_violation=safv, filter_active=filt, projection_residual=pres,
             projection_cost=pcost, distance_to_goal=dgoal, distance_to_obstacle=dobs, is_success=succ, time_to_goal=ttg, episode_return=epret)
    success_rate=float(np.mean(succ)); collision_rate=float(np.mean(np.any(dobs<0,axis=1))); far=float(np.mean(filt)); apr=float(np.mean(pres)); frr=float(np.mean(1-safv.astype(np.float32)))
    upper=routes.count('upper'); lower=routes.count('lower'); ns=max(len(routes),1); pu,pl=upper/ns,lower/ns
    route_entropy=float(-(pu*np.log(pu+1e-8)+pl*np.log(pl+1e-8)))
    summary=dict(success_rate=success_rate, collision_rate=collision_rate, filter_activation_rate=far, avg_projection_residual=apr, feasible_raw_action_ratio=frr,
                 route_upper_ratio=pu, route_lower_ratio=pl, route_entropy=route_entropy, episode_return_mean=float(np.mean(epret)), episode_return_std=float(np.std(epret)),
                 time_to_goal_mean=float(np.mean(ttg)), state_violation_rate=float(np.mean(sv)))
    (save_dir/'summary.json').write_text(json.dumps(summary, indent=2))

if __name__=='__main__': main()
