#!/usr/bin/python

states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for i in states:
        V[0][i] = start_p[i]*emit_p[i][obs[0]]
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for y in states:
            prob = max(V[t - 1][y0]*trans_p[y0][y]*emit_p[y][obs[t]] for y0 in states)
            V[t][y] = prob
    for i in dptable(V):
        print i
    opt = []
    for j in V:
      for x, y in j.items():
        if j[x] == max(j.values()):
          opt.append(x)
    # The highest probability
    h = max(V[-1].values())
    print 'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s'%h 

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%10d" % i) for i in range(len(V)))
    for y in V[0]:
        yield "%.7s: " % y+" ".join("%.7s" % ("%f" % v[y]) for v in V)

   