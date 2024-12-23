


from chemcrow.tools.retroplanner import RetroPlanner, RetroPlannerSingleStep

# print('#'*20)
# print('Multi Step Test')
# retroplanner_api = RetroPlanner()
# results = retroplanner_api._run('CC(=O)OC2CCCC(c1ccccc1)C2')
# print("results1:")
# print(results)
# results = retroplanner_api._run('a')
# print("results2:")
# print(results)
# results = retroplanner_api._run('C1CCCOCCCC(C=O)C1')
# print("results3:")
# print(results)

print('#'*20)
print('Single Step Test')
retroplanner_single_step_api = RetroPlannerSingleStep()
results = retroplanner_single_step_api._run('C1CCCOCCCC(C=O)C1')
print("results1:")
print(results)
pass