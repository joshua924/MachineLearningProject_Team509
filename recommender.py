import pickle
import numpy as np
import random


def loadModelAndFeature():
  print 'Loading the model and selected features ...'
  model = pickle.load(open('model/log_reg.model', 'r'))
  hero_map = {'Sand King': 16, 'Rubick': 86, 'Tinker': 34, 'Alchemist': 73, 'Luna': 48, 'Lina': 25, 
      'Earth Spirit': 107, 'Naga Siren': 89, 'Silencer': 75, 'Warlock': 37, 'Night Stalker': 60, 'Spectre': 67, 
      'Faceless Void': 41, 'Razor': 15, 'Arc Warden': 113, 'Bloodseeker': 4, 'Winter Wyvern': 112, 
      'Spirit Breaker': 71, "Nature's Prophet": 53, 'Lich': 31, 'Anti-Mage': 1, 'Tidehunter': 29, 'Beastmaster': 38, 
      'Slark': 93, 'Juggernaut': 8, 'Centaur Warrunner': 96, 'Viper': 47, 'Pugna': 45, 'Dark Seer': 55, 
      'Gyrocopter': 72, 'Phantom Lancer': 12, 'Nyx Assassin': 88, 'Witch Doctor': 30, 'Disruptor': 87, 
      'Lone Druid': 80, 'Tusk': 100, 'Clockwerk': 51, 'Abaddon': 102, 'Treant Protector': 83, 'Crystal Maiden': 5, 
      'Ursa': 70, 'Axe': 2, 'Elder Titan': 103, 'Riki': 32, 'Bane': 3, 'Storm Spirit': 17, 'Undying': 85, 'Chen': 66, 
      'Death Prophet': 43, 'Doom': 69, 'Jakiro': 64, 'Meepo': 82, 'Lycanthrope': 77, 'Mirana': 9, 'Chaos Knight': 81, 
      'Sniper': 35, 'Skeleton King': 42, 'Ember Spirit': 106, 'Phoenix': 110, 'Shadow Demon': 79, 'Clinkz': 56, 
      'Huskar': 59, 'Broodmother': 61, 'Abyssal Underlord': 108, 'Drow Ranger': 6, 'Oracle': 111, 'Visage': 92, 
      'Omniknight': 57, 'Necrophos': 36, 'Dazzle': 50, 'Tiny': 19, 'Keeper of the Light': 90, 'Earthshaker': 7, 
      'Bounty Hunter': 62, 'Shadow Fiend': 11, 'Pudge': 14, 'Shadow Shaman': 27, 'Puck': 13, 'Phantom Assassin': 44, 
      'Venomancer': 40, 'Medusa': 94, 'Troll Warlord': 95, 'Slardar': 28, 'Zeus': 22, 'Skywrath Mage': 101, 
      'Vengeful Spirit': 20, 'Morphling': 10, 'Brewmaster': 78, 'Queen of Pain': 39, 'Batrider': 65, 'Bristleback': 99, 
      'Timbersaw': 98, 'Templar Assassin': 46, 'Sven': 18, 'Wisp': 91, 'Enigma': 33, 'Kunkka': 23, 'Windranger': 21, 
      'Legion Commander': 104, 'Lion': 26, 'Lifestealer': 54, 'Terrorblade': 109, 'Weaver': 63, 'Magnus': 97, 
      'Outworld Devourer': 76, 'Dragon Knight': 49, 'Ogre Magi': 84, 'Ancient Apparition': 68, 'Enchantress': 58, 
      'Invoker': 74, 'Techies': 105, 'Leshrac': 52}
  features = np.loadtxt('feature_selected/selected_features.txt', delimiter=',', dtype='S')
  feature_dict = {}
  for i, each in enumerate(features):
    feature_dict[each] = i
  print 'Done'
  return model, feature_dict, hero_map


'''
add new features to the current feature list.

@param feature_list: the current feature list
@param dire: list of current dire heroes
@param radiant: list of current radiant heroes
@param new_pick: a string indicating the hero id to be picked
@param side: "d" if the new pick will be added to dire side, otherwise "r"
'''
def generate_new_feature_vector(feature_list, dire, radiant, new_pick, side):
  if side == 'd':
    feature_list.append('d_%s' % new_pick)
    for each in dire:
      feature_list.append('d_%s_d_%s' % (new_pick, each))
      feature_list.append('d_%s_d_%s' % (each, new_pick))
    for each in radiant:
      feature_list.append('d_%s_r_%s' % (new_pick, each))
    dire.append(new_pick)
  else:
    feature_list.append('r_%s' % new_pick)
    for each in dire:
      feature_list.append('d_%s_r_%s' % (each, new_pick))
    for each in radiant:
      feature_list.append('r_%s_r_%s' % (new_pick, each))
      feature_list.append('r_%s_r_%s' % (each, new_pick))
    radiant.append(new_pick)


def make_vector(feature_list, feature_dict):
  vec = np.zeros(len(feature_dict))
  for each in feature_list:
    if each in feature_dict:
      vec[feature_dict[each]] = 1
  return vec


def printProb(classes, probs):
  print '-' * 40 + 'winning probability'+ '-' * 41
  for i, each in enumerate(classes):
    tag = 'dire' if each == -1 else 'radiant'
    tag += ': %-6.4f' % probs[i]
    print tag.center(100)
  print '-' * 100


def print_sides(a, b, dic):
  print '-' * 47 + 'lineup' + '-' * 47
  print ('dire: %s' % [dic[t] for t in a]).center(100)
  print ('radiant: %s' % [dic[t] for t in b]).center(100)
  print '-' * 100


def reverseMap(hero_map):
  dic1, dic2 = {}, {}
  for k, v in hero_map.items():
    dic1[v] = k
    dic2[v] = k
  return dic1, dic2


def simulate():
  model, feature_dict, hero_map = loadModelAndFeature()
  id_hero_map, lookup = reverseMap(hero_map)
  print "The engine will pick on behalf of Radiant, and "\
      + "you will pick for Dire, and you will go first. Ready? Let's go!"
  dire, radiant, feature_list = [], [], []
  for i in range(5):
    choice = raw_input('What is your pick for Dire: ')
    while choice not in hero_map or hero_map[choice] not in id_hero_map:
      print "No hero named %s or it's already picked, please check!" % choice
      choice = raw_input('What is your pick: ')
    generate_new_feature_vector(feature_list, dire, radiant, hero_map[choice], 'd')
    del id_hero_map[hero_map[choice]]
    
    max_prob, max_idx = 0.0, -1
    for i in id_hero_map:
      fake_list, fake_d, fake_r = list(feature_list), list(dire), list(radiant)
      generate_new_feature_vector(fake_list, fake_d, fake_r, i, 'r')
      ins = make_vector(fake_list, feature_dict)
      dire_prob = model.predict_proba(ins.reshape(1, -1))[0][1]
      if dire_prob > max_prob:
        max_prob = dire_prob
        max_idx = i
      elif dire_prob == max_prob:
        max_idx = [max_idx, i][random.randint(0, 1)]
    print 'Recommendation engine picks %s for Radiant' % id_hero_map[max_idx]
    generate_new_feature_vector(feature_list, dire, radiant, max_idx, 'r')
    ins = make_vector(feature_list, feature_dict)
    print '%s features hit' % sum(ins)
    print_sides(dire, radiant, lookup)
    printProb(model.classes_, model.predict_proba(ins.reshape(1, -1))[0])
    del id_hero_map[max_idx]


if __name__ == '__main__':
  simulate()

