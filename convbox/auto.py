msdlayer = 'MSDNet_Layer5_2'
'''
for i in range(4):
 if i==0:
  for j in range(6):
    if j==0 or j==3:
      print msdlayer+'['+str(i)+']'+'.modules[0].modules[1].modules['+str(j)+'].weight'
    elif j==1 or j==4:
      print msdlayer+'['+str(i)+']'+'.modules[0].modules[1].modules['+str(j)+'].bias'
      print msdlayer+'['+str(i)+']'+'.modules[0].modules[1].modules['+str(j)+'].weight'
    else:
      continue
 else:
  for k in range(2):
    for j in range(6):
       if j==0 or j==3:
         print msdlayer+'['+str(i)+']'+'.modules[0].modules['+str(k+1)+'].modules['+str(j)+'].weight'
       elif j==1 or j==4:
         print msdlayer+'['+str(i)+']'+'.modules[0].modules['+str(k+1)+'].modules['+str(j)+'].bias'
         print msdlayer+'['+str(i)+']'+'.modules[0].modules['+str(k+1)+'].modules['+str(j)+'].weight'
       else:
         continue 




for i in range(2):
  for k in range(2):
    for j in range(6):
       if j==0 or j==3:
         print msdlayer+'['+str(i)+']'+'.modules[0].modules['+str(k+1)+'].modules['+str(j)+'].weight'
       elif j==1 or j==4:
         print msdlayer+'['+str(i)+']'+'.modules[0].modules['+str(k+1)+'].modules['+str(j)+'].bias'
         print msdlayer+'['+str(i)+']'+'.modules[0].modules['+str(k+1)+'].modules['+str(j)+'].weight'
       else:
         continue 


for i in range(3):
 if i==0:
  for j in range(6):
    if j==0 or j==3:
      print msdlayer+'['+str(i)+']'+'.modules[0].modules[1].modules['+str(j)+'].weight'
    elif j==1 or j==4:
      print msdlayer+'['+str(i)+']'+'.modules[0].modules[1].modules['+str(j)+'].bias'
      print msdlayer+'['+str(i)+']'+'.modules[0].modules[1].modules['+str(j)+'].weight'
    else:
      continue
 else:
  for k in range(2):
    for j in range(6):
       if j==0 or j==3:
         print msdlayer+'['+str(i)+']'+'.modules[0].modules['+str(k+1)+'].modules['+str(j)+'].weight'
       elif j==1 or j==4:
         print msdlayer+'['+str(i)+']'+'.modules[0].modules['+str(k+1)+'].modules['+str(j)+'].bias'
         print msdlayer+'['+str(i)+']'+'.modules[0].modules['+str(k+1)+'].modules['+str(j)+'].weight'
       else:
         continue 
'''


for i in range(1):
  for j in range(6):
    if j==0 or j==3:
      print msdlayer+'['+str(i)+']'+'.modules[0].modules[1].modules['+str(j)+'].weight'
    elif j==1 or j==4:
      print msdlayer+'['+str(i)+']'+'.modules[0].modules[1].modules['+str(j)+'].bias'
      print msdlayer+'['+str(i)+']'+'.modules[0].modules[1].modules['+str(j)+'].weight'
    else:
      continue
