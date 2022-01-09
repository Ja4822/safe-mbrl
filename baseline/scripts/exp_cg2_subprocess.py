import subprocess


def experiment_button(algo):
    task_list = ['goal2']
    algo_list = ['cpo']

    robot = 'car'

    p_list = []
    for task in task_list:
        for algo in algo_list:
            for cost in [2]:
                for seed in [0, 11, 22]:
                    print("Robot: ", robot, " Algo: ", algo, " Seed: ", seed, " Cost: ", cost)
                    command = ['python'] + ['scripts/experiment.py'] + ['--robot'] + [robot] + ['--task'] + [task] + \
                        ['--algo'] + [algo] + ['--cost'] + [str(cost)] + ['--seed'] + [str(seed)]
                    p = subprocess.Popen(command)
                    p_list.append(p)

                global pid_list
                pid_list = [p.pid for p in p_list]

                for p in p_list:
                    p.wait()

                import os
                import signal
                def kill_child():
                    global pid_list
                    for pid in pid_list:
                        if pid is None:
                            pass
                        else:
                            try:
                                os.kill(pid, signal.SIGTERM)
                            except OSError:
                                pass
                            else:
                                pass

                import atexit
                atexit.register(kill_child)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo','-a', type=str, default='cpo')
    args = parser.parse_args()
    experiment_button(args.algo)