########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Notes: This code is provided without warranty.

### Python data (scalar) types
ival = 17239871
ival ** 6


print (type(ival))



3 / 2



print (type(3/2)) # 'float'


cval = 1 + 2j
print (cval * (1 - 2j))
print (type(cval))



False or True



a = [1, 2, 3]
if a:
    print ('I found something!')



b = []
if not b:
    print ('Empty!')



bool('Hello world!'), bool('')



bool(0), bool(1)



s = '3.14159'
fval = float(s)
type(fval)



int(fval)



bool(fval)



bool(0)



a = None
a is None



b = 5
b is not None



def add_and_maybe_multiply(a, b, c=None):
    result = a + b
    if c is not None:
        result = result * c
    return result



add_and_maybe_multiply(2, 3)



add_and_maybe_multiply(2, 3, 10)



from datetime import datetime, date, time



dt = datetime(2011, 10, 29, 20, 30, 21)
type(dt)
dt

dir(dt) # Query methods which object 'dt' can use.
[(name,type(getattr(dt,name))) for name in dir(dt)] # https://stackoverflow.com/questions/26818007/python-dir-how-can-i-differentiate-between-functions-method-and-simple-att

dt.day



dt.minute



dt.date()



dt.time()



dt.strftime('%m/%d/%Y %H:%M')



datetime.strptime('20091031', '%Y%m%d')



dt.replace(minute=0, second=0) # replacing the minute and second fields with zero



dt2 = datetime(2011, 11, 15, 22, 30)
delta = dt2 - dt
delta



type(delta)



dt + delta



### Strings Fundamentals
a = 'this is a string'
type(a)


### immutable
print (a[10])
a[10] = 'f' # TypeError: 'str' object does not support item assignment

a = 5.6
s = str(a)
s



type(s)

r'Hel\nlo'

'Hel\nlo'

s = '12\\34'
print(s)



s = r'this\has\no\special\characters'
print(s)

# Normal string with escape sequences leading to a wrong file path!
escaped_string = "C:\the_folder\new_dir\file.txt"
print (escaped_string) # will cause errors if we try to open a file, because \t, \n, and \f here

# raw string keeping the backslashes in its normal form
raw_string = r'C:\the_folder\new_dir\file.txt'
print (raw_string)

# unicode string literals
string_with_unicode = u'H\u00e8llo!'
print (string_with_unicode)
ord(string_with_unicode)
# TypeError: ord() expected a character, but string of length 5 found

string_with_unicode = u'H\xe8llo'
print (string_with_unicode)
ord(string_with_unicode)
# TypeError: ord() expected a character, but string of length 5 found


a = 'this is the first half '
b = 'and this is the second half'
a + b

simple_string = 'hello' + " I'm a simple string"
simple_string


# https://unicode-table.com/cn/2639/
smiley = u"\u263A"

print(smiley)

type(smiley) # str

(u"\u263A").decode()
# AttributeError: 'str' object has no attribute 'decode'

ord(smiley) # = 9786 decimal value of code point = 263A in Hex. (Return the Unicode code point for a one-character string.)

len(smiley)

smiley.encode('utf8') # prints '\xe2\x98\xba' the bytes - it is <str>

type(smiley.encode('utf8')) # bytes

len(smiley.encode('utf8')) # its length = 3, it means 3 bytes

print (b'\xe2\x98\xba')

(b'\xe2\x98\xba').decode()

print(u"\u263A".encode('ascii')) # 'ascii' codec can't encode character '\u263a' in position 0: ordinal not in range(128)


template = '%.2f %s are worth $%d'
template % (31.5560, 'Taiwan Dollars', 1)

# multi-line string, note the \n (newline) escape character automatically created
multi_line_string = """Hello I'm
a multi-line
string!"""
multi_line_string
# Attention to the difference to above
print (multi_line_string)


### Python native data structures
tup = 4, 5, 6
tup



nested_tup = (4, 5, 6), (7, 8)
nested_tup
`

tuple([4, 0, 2])



tup = tuple('string')
tup



tup[0]



tup = tuple(['foo', [1, 2], True])
tup
# tup[2] = False # TypeError: 'tuple' object does not support item assignment



tup[1].append(3)
tup



(4, None, 'foo') + (6, 0) + ('bar',)



('foo', 'bar') * 4



tup = (4, 5, 6)
a, b, c = tup
c



x, y = 1, 2 # now x is 1, y is 2
x, y = y, x # Pythonic way to swap variables; now x is 2, y is 1
x



y



a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)



a.index(3)



c



a_list = [2, 3, 7, None]
a_list[1] = 'peekaboo'
a_list

dir(a_list)
help(a_list.append)

a_list.append('dwarf')
a_list



a_list.insert(1, 'red')
a_list



a_list.pop(1)



a_list



a_list.append('peekaboo')
a_list



a_list.remove('peekaboo')
a_list



'dwarf' in a_list



[4, None, 'foo'] + [7, 8, (2, 3)]
`

x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])
x


import random
everything = []
list_of_lists = np.random.standard_normal((5000, 5000))
import time
start = time.time()
for chunk in list_of_lists:
    everything.extend(chunk)
end = time.time()
print('The elapsed time is: {}'.format(end-start)) # 0.22289514541625977

everything = []
list_of_lists = np.random.standard_normal((1000, 1000))
start = time.time()
for i, chunk in enumerate(list_of_lists):
    if i == 0:
        everything = chunk.tolist()
    else:
        everything = everything + chunk.tolist()
end = time.time()
print('The elapsed time is: {}'.format(end-start)) # 5.242393732070923


a = [7, 2, 5, 1, 3]
a.sort()
a

a.sort(reverse = True)
a


b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)
b



import bisect
c = [1, 2, 2, 2, 3, 4, 7]
bisect.bisect(c, 2)



bisect.bisect(c, 5)



bisect.insort(c, 6)
c



seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[1:5]



seq[3:4] = [6, 3]
seq



seq[:5]



seq[3:]



seq[-4:]



seq[-6:-2]



seq[::2]



seq[::-1]


# dict
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
d1



d1[7] = 'an integer'
d1

d1['b']

d1['c'] = d1.pop(8)

'b' in d1



d1[5] = 'some value'
d1['dummy'] = 'another value'
d1



del d1[5]
d1



ret = d1.pop('dummy')
ret



print (d1.keys())
d1.values()



d1.update({'b' : 'foo', 'c' : 12})
d1



mapping = dict(zip(range(5), reversed(range(5))))
mapping



words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)

by_letter



words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    by_letter.setdefault(letter, []).append(word)

by_letter



from collections import defaultdict



by_letter = defaultdict(list)
by_letter



for word in words:
    by_letter[word[0]].append(word)



by_letter



hash('string')



hash((1, 2, (2, 3)))



# hash((1, 2, [2, 3])) # fails because lists are mutable



d = {}
d[tuple([1, 2, 3])] = 5
d


# set
set([2, 2, 2, 1, 3, 3])



{2, 2, 2, 1, 3, 3}



a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
a | b # union (or)



a & b # intersection (and)



a - b # difference



a ^ b # symmetric difference (xor)



a_set = {1, 2, 3, 4, 5}
{1, 2, 3}.issubset(a_set)



a_set.issuperset({1, 2, 3})



{1, 2, 3} == {3, 2, 1}

### Operations and Operators
1 + 1 == 2
1 + 1 is 2
8 + 7 == 87
8 + 9 is not 1
"放生" == "棄養"
False or True
False and True
False | True
False & True
bool('Hello world!')
bool('')
bool(0)
bool(1)

USD = 31.3987
JPY = 0.2738
USD * JPY + JPY/USD + USD **2 + JPY ** (1/2)

JPY ** (1/2)

### control statements
x = -99
if x < 0:
    print ('It is negative')



x = 99
if x < 0:
    print ('It is negative')
elif x == 0:
    print ('Equal to zero')
elif 0 < x < 5:
    print ('Positive but smaller than 5')
else:
    print ('Positive and larger than or equal to 5')



x=0
while x < 5:
    print (x, "is less than 5")
    x += 1



for x in range(5):
    print (x, "is less than 5")



for x in range(10):
    if x==3:
        continue # go immediately to the next iteration
    if x==5:
        break # quit the loop entirely
    print (x)


### User-defined functions
def my_function(x, y, z=1.5):
    if z > 1:
        return z * (x + y)
    else:
        return z / (x + y)



my_function(5, 6, z=0.7)



my_function(3.14, 7, 3.5)



def func():
    a = []
    for i in range(5):
        a.append(i)



func() # return nothing because of local scoping
# a # NameError: name 'a' is not defined



a = []
def func():
    for i in range(5):
        a.append(i)



func()
a



a


### Python 物件導向觀念 object-oriented concept
class Bird(object):
	have_feather = True
	way_of_reproduction = 'egg'



summer = Bird()
print (summer.way_of_reproduction)
dir(summer)



class Bird(object):
	have_feather = True
	way_of_reproduction = 'egg'
	def move(self, dx, dy): # 參數中有一個self，是為了方便我們引用對象自身。方法的第一個參數必須是self，無論是否用到 !!!
		position = [0,0]
		position[0] = position[0] + dx
		position[1] = position[1] + dy
		return position



summer = Bird()
print ('after move:', summer.move(5,8))
dir(summer)
[(name,type(getattr(summer,name))) for name in dir(summer)]


class Chicken(Bird): # 繼承自何物件寫在括弧裡，另外再新增兩屬性
	way_of_move = 'walk'
	possible_in_KFC = True
	
class Oriole(Bird):
	way_of_move = 'fly'
	possible_in_KFC = False

summer = Chicken()
dir(summer)
print (summer.have_feather) # 繼承自父類的變項屬性
print (summer.move(5,8)) # 繼承自父類的方法屬性



class Human(object):
	laugh = 'hahahaha' # an attribute of class Human
	def show_laugh(self): # 參數中有一個self，是為了方便我們引用物件本身。方法的第一個參數必須是self，無論是否用到!!!
		print (self.laugh) # 在定義方法時，必須有self參數。這個參數表示某個擁有類別所有屬性的物件，如此我們可以透過self.xxx，
	def laugh_10th(self): # 調用類別屬性xxx，或是方法xxx
		for i in range(10):
			self.show_laugh()

vince = Human()
vince.laugh_10th()



class happyBird(Bird): # 類別Bird要先定義, 因為happyBird繼承Bird
	def __init__(self, more_words): # 創建時會自動呼叫的物件初始化函數
		print ('We are happy birds.', more_words)

summer = happyBird('Happy, Happy!') # 要看初始化函數如何運作！而非class happyBird(Bird)



class Human(object):
	def __init__(self, input_gender):
		self.gender = input_gender # it's an ***object*** attribute, NOT a class atrribute !
	def printGender(self):
		print (self.gender) # use the gender attribute of object itself



vince = Human('male') # 'males'是初始化函數中接收的input_gender，透過初始化特殊方法給予物件屬性值
print (vince.gender)
vince.printGender() # same as above

dir(Human) # 沒有屬性gender，只有printGender方法
Human.__class__



### 其他觀念 Other concepts
# list comprehension

#[expr for val in collection if condition]

#result = []
#for val in collection:
#    if condition:
#        result.append(expr)

strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]


# dict comprehension
unique_lengths = {len(x) for x in strings}
unique_lengths



print (enumerate(strings))
print (list(enumerate(strings)))
loc_mapping = {val : index for index, val in enumerate(strings)}
loc_mapping


# 另一種寫法！
loc_mapping = dict((val, idx) for idx, val in enumerate(strings)) # same as above
loc_mapping



x=None
print (x == None) # prints True, but is not Pythonic print x is None
print (x is None) # prints True, and is Pythonic



s = "abc" # Try 123 (not "123"): TypeError: 'int' object is not subscriptable
if s:
    first_char = s[0]
else:
    first_char = ""

first_char



first_char = s and s[0] # same as above
first_char



x = 123
safe_x = x or 0
safe_x



all([True, 1, { 3 }])



all([True, 1, {}]) # False, {} is falsy



any([True, 1, {}]) # True, True is truthy



all([]) # True, no falsy elements in the list, intreresting!



any([]) # False, no truthy elements in the list, very intreresting! (所以[]是沒有任何真假元素在其中!!!)


### Generators and Iterators
def lazy_range(n):
    """a lazy version of range"""
    i=0
    while i < n:
        yield i
        i += 1



for i in lazy_range(10):
    print (i + 1000)



def natural_numbers():
    """returns 1, 2, 3, ..."""
    n=1
    while True:
        yield n
        n+=1


# 另一種建立generators的方法是包在小括弧裡for推導
lazy_evens_below_20 = (i for i in lazy_range(20) if i % 2 == 0)
print (lazy_evens_below_20)
list(lazy_evens_below_20)


### Python looping
for i in range(4):
    print (i)



S = 'abcd'
for (index, char) in enumerate(S):
    print (index, char)



ta = [1,2,3]
tb = [9,8,7]
tc = ['a','b','c']
for (a,b,c) in zip(ta,tb,tc):
    print (a,b,c)


### Functional tools
def exp(base, power):
    return base ** power



def two_to_the(power):
    return exp(2, power)



from functools import partial
two_to_the = partial(exp, 2) # is now a function of one variable
print (two_to_the(3)) # 8



def double(x):
    return 2*x

xs=[1,2,3,4]
twice_xs = [double(x) for x in xs]
twice_xs



twice_xs = map(double, xs)
print (twice_xs)
list(twice_xs)



list_doubler = partial(map, double)
twice_xs = list_doubler(xs)
print (twice_xs)
list(twice_xs)



def multiply(x, y): return x * y
products = map(multiply, [1, 2], [4, 5]) # [1 * 4, 2 * 5] = [4, 10]
list(products)



def is_even(x):
    """True if x is even, False if x is odd"""
    return x % 2 == 0

x_evens = [x for x in xs if is_even(x)] # [2, 4]
x_evens



x_evens = filter(is_even, xs) # same as above
print (x_evens)
list(x_evens)



list_evener = partial(filter, is_even) # *function* that filters a list
x_evens = list_evener(xs) # again [2, 4]
print (x_evens)
list(x_evens)



xs



import functools as ft # Python 3
x_product = ft.reduce(multiply, xs)
x_product

# [(name, type(getattr(ft, name))) for name in dir(ft)]

list_product = partial(ft.reduce, multiply)
x_product = list_product(xs)
x_product # same as above
