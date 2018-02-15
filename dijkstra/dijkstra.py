import queue
import sys

mygraph = {'A': [('D',8), ('C',3)],
            'C': [('A',3), ('F',6), ('E',1)],
            'D': [('A',8), ('E',2)],
            'E': [('D',2), ('B',2), ('C',1)],
            'B': [('E',2)],
            'F': [('C',6)]}

#relative infinity
inf = 99


def dijkstras(g, source, iterations):
    #initialize root
    dist = {source: 0}

    for k,v in g.items():
        for j in v:
            if j[0] not in dist:
                dist[j[0]] = inf

    for i in range(iterations):
        for k in dist:
            v = g[k]

            for u in v:
                alt = dist[k] + u[1]
                if alt < dist[u[0]]:
                    dist[u[0]] = alt
                    print("updated: " + str(dist))

    return dist


def main():
    iterations = 2  if len(sys.argv) < 2  else (int(sys.argv[1]))

    for k,v in mygraph.items():
        print('source:' + k +': ' )
        print(str(dijkstras(mygraph, k, iterations)))

if __name__ == '__main__':
    main()
