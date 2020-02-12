
import numpy as np, json

nprods = 10
def rand(l, u , n, r):
    if n == 1:
        return np.random.uniform(l,u,n).round(r)[0]
    else:
        return list(np.random.uniform(l,u,n).round(r))

data = {'price': rand(50,80,nprods,2),
        'weight_min': rand(10,40,nprods,2),
        'weight_max': rand(70,100,nprods,2),
        'facility_capacity': {'B': rand(80,120,1,2),'C':rand(80,120,1,2) ,'D':rand(80,120,1,2) ,'E':rand(200,250,1,2)},
        'facility_cost': {'B': rand(1,5,1,2),'C':rand(1,5,2,2) ,'D': [rand(12,13,1,2),rand(13,14,1,2),rand(14,15,1,2)]},
        'transportation_capacity': {1: {1: rand(80,120,1,2), 2: rand(80,120,1,2), 3: rand(80,120,1,2)}, 2: {1: rand(80,120,1,2), 2: rand(80,120,1,2), 3: rand(80,120,1,2), 4: rand(80,120,1,2)}},
        'transportation_cost': {1: {1: rand(1,5,1,2), 2: rand(1,5,1,2), 3: rand(1,5,1,2)}, 2: {1: rand(1,5,1,2), 2: rand(1,5,1,2), 3: rand(1,5,1,2), 4: rand(1,5,1,2)}},
        'eps': 1e-3}

data['weight_threshold'] = {'C': 0.5 * data['facility_capacity']['C'],'D': [0.33 * data['facility_capacity']['D'], 0.66 * data['facility_capacity']['D']]}

with open('../solving_MINLP_in_CVXPY/data.json', 'w') as dat:
    json.dump(data, dat)

