import pickle

new_labelset =pickle.load(open('/data0/wwang/pll_0507/labelset_rafdb/new_labelset.pkl', 'rb'))
a22 =pickle.load(open('/data0/wwang/pll_0507/labelset_rafdb/113_54.pkl', 'rb'))
count=0
for k, v in a22.items():
    if k in new_labelset:
        print(k)
        print('a22_{}'.format(v))
        print('new_labelset_{}'.format(new_labelset[k]))
        count+=1
        print(count)

pass