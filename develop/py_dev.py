import os, sys 

# 1. Testing for getattr
# Get the attribute value from a class

class A(object):
    bar = 1
    
a = A
b = getattr(a, "bar")
#print(b) # 1


# 2. Testing  __setitem__, __getitem__ 
# make the class as a dictionary

class DictDemo:
    def __init__(self,key,value):
        self.dict = {}
        self.dict[key] = value
    def __getitem__(self,key):
        return self.dict[key]
    def __setitem__(self,key,value):
        self.dict[key] = value
    def __len__(self):
        return len(self.dict)
dictDemo = DictDemo('key0','value0')
#print(dictDemo['key0']) #value0
dictDemo['key1'] = 'value1'
#print(dictDemo['key1']) #value1
#print(len(dictDemo)) #2
#print(dictDemo.dict)  #{'key0': 'value0', 'key1': 'value1'}


# 3. Testing the __call__ function
# Make the class as a callable object

class A(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def __call__(self, c):
        #print("The value is", c)
        return c

obj = A("a", "b")
x = obj("c")
#print(x)
