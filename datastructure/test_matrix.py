from numpy import * 
a = array([['Mon',18,20,22,17],['Tue',11,18,21,18],
		   ['Wed',15,21,20,19],['Thu',11,20,22,21],
		   ['Fri',18,17,23,22],['Sat',12,22,20,18],
		   ['Sun',13,15,19,16]])
    
print(a)

m = reshape(a,(7,5))
print(m)


m_r = append(m,[["app1","app2","app3","app4","app5"]],0)
print(m_r)

m_rr = append(m,[["day",0,1,2,3]],0)
print(m_rr)

m_c = insert(a,[5],[[1],[2],[3],[4],[5],[6],[7]],1)
print(m_c)

r_r = delete(a,[-1],0)
print(r_r)

r_c = delete(a,[-1],1)
print(r_c)

