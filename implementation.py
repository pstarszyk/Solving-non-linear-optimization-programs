
import cvxpy as cp, numpy as np, json, datetime

# Load problem set data

with open('../solving_MINLP_in_CVXPY/data.json') as data:
    data = json.load(data)

# define variables
nw = len(data['price'])
nz = nw * 5 # recall that routes 1,2 on day 2 are modeled with the same variables as routes 1,2 from day 1
nX = nw * 5 

w = cp.Variable(nw)
z = cp.Variable(nX, boolean=True) # (15)
X = cp.Variable(nX)
y = cp.Variable(4, boolean=True) # (15)

# Set input structures: define matrices for summations

# array for (1')
v1 = data['price']*3

# array for (4)
v4 = np.zeros(4)
v4[0] = data['eps']
v4[1] = data['facility_cost']['D'][0]
v4[2] = np.sum(data['facility_cost']['D'][0:2])
v4[3] = np.sum(data['facility_cost']['D'])

# array for (5')
v5 = [data['transportation_cost'][1][1]]*nw + [data['transportation_cost'][1][2]]*nw + [data['transportation_cost'][1][3]]*nw

# array for (6')
v6a = [data['transportation_cost'][2][1]]*nw + [data['transportation_cost'][2][2]]*nw
v6b = [data['transportation_cost'][2][3]]*nw + [data['transportation_cost'][2][4]]*nw

# minimum values for (7') and (8')
T_min_1 = np.min((data['transportation_capacity'][1][1], data['transportation_capacity'][2][1]))
T_min_2 = np.min((data['transportation_capacity'][1][2], data['transportation_capacity'][2][2]))
T_min_3 = np.min((data['transportation_capacity'][1][3], data['transportation_capacity'][2][3] + data['transportation_capacity'][2][4]))

# matrix for (10)
M10 = np.zeros((nw*2, nw*5))
for rr in range(2):
    for cc in range(5):
        if (rr,cc) in [(0,0), (0,1), (0,2), (1,0), (1,1), (1,3), (1,4)]:
            M10[nw*rr:nw*(rr+1), nw*cc:nw*(cc+1)] = np.eye(nw)

# matrix for (11')
M11 = np.kron(np.eye(3), np.ones(nw))
S_max = list(data['facility_capacity'].values())[0:3]

# matrix for (12)
M12 = np.c_[np.eye(nw), np.eye(nw)]

# array for (16)
L = [0, data['eps']] + data['weight_threshold']['D']
U = [data['eps']] + data['weight_threshold']['D'] + [data['facility_capacity']['D']]

# array for (18) and (19)
w_max = data['weight_max']*5
w_min = data['weight_min']*5

# matrix for (20)
M20 = np.kron(np.ones(5).reshape(-1,1), np.eye(nw))

def objective_function(w, z, X, y):

    revenue = -1 * cp.sum(cp.multiply(X[0:nw*3], v1)) # (1')
    
    storage_cost_B = cp.multiply(data['facility_cost']['B'], cp.sum(X[0:nw])) # (2')
    storage_cost_C = cp.multiply(data['facility_cost']['C'][0], cp.pos(cp.sum(X[nw:nw*2]) - data['weight_threshold']['C'])) \
                     + cp.multiply(data['facility_cost']['C'][0], cp.sum(z[nw:nw*2]))  # (3')
    storage_cost_D = cp.sum(cp.multiply(v4, y)) # (4)
    
    transportation_cost_D1 = cp.sum(cp.multiply(X[0:nw*3], v5)) # (5')
    transportation_cost_D2 = cp.sum(cp.multiply(X[0:nw*2], v6a)) + cp.sum(cp.multiply(X[nw*3:], v6b)) # (6')

    return revenue + storage_cost_B + storage_cost_C + storage_cost_D + transportation_cost_D1 + transportation_cost_D2


constraints = [cp.sum(X[0:nw]) <= T_min_1, # (7') j=1
               cp.sum(X[nw:nw*2]) <= T_min_2, # (7') j=2
               cp.sum(X[nw*2:nw*3]) <= T_min_3, # (8')
               cp.sum(X[nw*3:nw*4]) <= data['transportation_capacity'][2][3], # (9') j = 3
               cp.sum(X[nw*4:]) <= data['transportation_capacity'][2][4], # (9') j = 4
               cp.matmul(M10, z) <= 1, # (10)
               cp.matmul(M11, X[0:nw*3]) <= S_max, # (11')
               cp.matmul(M12, z[nw*3:]) >= z[nw*2:nw*3], # (12)
               cp.matmul(M12, z[nw*3:]) <= z[nw*2:nw*3], # (12)
               cp.sum(X[0:nw*3]) <= data['facility_capacity']['E'], # (13)
               w >= data['weight_min'], # (14)
               w <= data['weight_max'], # (14)
               cp.sum(X[nw*2:nw*3]) >= cp.sum(cp.multiply(L, y)), # (16)
               cp.sum(X[nw*2:nw*3]) <= cp.sum(cp.multiply(U, y)), # (16)
               cp.sum(y) >= 1, # (17)
               cp.sum(y) <= 1, # (17)
               X >= 0, # (18)
               X <= w_max, # (18)
               X >= cp.multiply(z, w_min), # (19)
               X <= cp.multiply(z, w_max), # (19)
               X >= cp.matmul(M20, w) - cp.multiply((1 - z), w_max), # (20)
               X <= cp.matmul(M20, w) - cp.multiply((1 - z), w_min), # (20)
               X <= cp.matmul(M20, w) + cp.multiply((1 - z), w_max)] # (21)


objective = cp.Minimize(objective_function(w, z, X, y))
prob = cp.Problem(objective, constraints)

start_time = datetime.now()
prob.solve(solver = cp.GLPK_MI)
end_time = datetime.now()


# Summary

def summary():
    w_opt = w.value
    z_opt = z.value
    X_opt = X.value
    y_opt = y.value
    
    X_shipped = np.matmul(np.kron(np.ones(3).reshape(-1,1), np.eye(nw)).T, X_opt[0:3*nw])
    z_stored = np.zeros((3, nw))
    z_stored[0,:] = z_opt[0:nw]
    z_stored[1,:] = z_opt[nw:2*nw]
    z_stored[2,:] = z_opt[2*nw:3*nw]
    
    print('Runtime: ' + str((end_time - start_time).microseconds) + ' microseconds.')

    for ii in range(nw):
        if X_shipped[ii] != 0:
            facility_ix = list(z_stored[:,ii]).index(1)
            facility = 'B' * (facility_ix == 0) + 'C' * (facility_ix == 1) + 'D' * (facility_ix == 2)
            print('Product ' + str(ii+1) + ' shipment: ' + str(np.round(X_shipped[ii], 2)) + ' tonnes stored at facility ' + facility + '.')
    print('Total Profit: $' + str(-np.round(prob.value, 2)))
    


summary()

