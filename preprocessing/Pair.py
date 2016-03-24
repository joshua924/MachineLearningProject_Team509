f = open('match.csv', 'r')
g = open('player.sql', 'r')
h = open('match_player.csv', 'w')

f.readline()
match_player = {}
radiant_win = {}

for line in f.readlines():
  mid, radiantwin = line.split(',')[0], line.split(',')[1]
  match_player[mid] = []
  radiant_win[mid] = 'radiant' if radiantwin == '1\n' else 'dire'
f.close()

for line in g.readlines():
  li = line.replace("'", "").split(", ")
  p = li[0].split('(')[1], li[4], li[-2]
  if p[0] in match_player:
  	match_player[p[0]].append((p[1], p[2]))
g.close()

h.write('winside,d_1,d_2,d_3,d_4,d_5,r_1,r_2,r_3,'+
  'r_4,r_5\n')
for key, value in match_player.iteritems():
  if len(value) == 10:
    h.write(radiant_win[key] + ',')
    value.sort(key=lambda p : p[1])
    if radiant_win[key] == 'dire':
      value.reverse()
    for hero, win in value[:-1]:
    	h.write(hero + ",")
    h.write(value[-1][0] + '\n')
h.close()