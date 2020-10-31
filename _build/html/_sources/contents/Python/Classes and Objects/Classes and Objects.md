# Classes and Objects

```{admonition} Original Source:
:class: tip
[https://www.geeksforgeeks.org/python-classes-and-objects/](https://www.geeksforgeeks.org/python-classes-and-objects/)
```

## Class

A class is a user-defined blueprint or prototype from which objects are created. Classes provide a means of bundling data and functionality together. Creating a new class creates a new type of object, allowing new instances of that type to be made. Each class instance can have attributes attached to it for maintaining its state. Class instances can also have methods (defined by its class) for modifying its state.

To understand the need for creating a class let’s consider an example, let’s say you wanted to track the number of dogs which may have different attributes like breed, age. If a list is used, the first element could be the dog’s breed while the second element could represent its age. Let’s suppose there are 100 different dogs, then how would you know which element is supposed to be which? What if you wanted to add other properties to these dogs? This lacks organization and it’s the exact need for classes.

Class creates a user-defined data structure, which holds its own data members and member functions, which can be accessed and used by creating an instance of that class. A class is like a blueprint for an object.

Some points on Python class:

- Classes are created by keyword `class`.
- Attributes are the variables that belong to class.
- Attributes are always public and can be accessed using dot (.) operator. Eg.: Myclass.Myattribute

## Objects

An Object is an instance of a Class. A class is like a blueprint while an instance is a copy of the class with actual values. It’s not an idea anymore, it’s an actual dog, like a dog of breed pug who’s seven years old. You can have many dogs to create many different instances, but without the class as a guide, you would be lost, not knowing what information is required.

An object consists of :

- State : It is represented by attributes of an object. It also reflects the properties of an object.
- Behavior : It is represented by methods of an object. It also reflects the response of an object with other objects.
- Identity : It gives a unique name to an object and enables one object to interact with other objects.

```{figure} ./image1.png
---
height: 150px
name: image1
---
```

### Declaring Objects (Also called instantiating a class)

When an object of a class is created, the class is said to be instantiated. All the instances share the attributes and the behavior of the class. But the values of those attributes, i.e. the state are unique for each object. A single class may have any number of instances.

```{figure} ./image2.png
---
height: 250px
name: image2
---
```

````{panels}
```
class Dog:
    
    '''Attributes of the class'''
    attribute1 = "Mammal"
    attribute2 = "Dog"
    
    '''Functions'''
    def who_am_i(self):
        print("I am a "+ self.attribute1)
        print("I am a "+ self.attribute2)
        
'''Creating an object of the class'''    
Tommy = Dog()

'''Accessing the characteristics of Tommy'''

print(Tommy.attribute1)
print(Tommy.attribute2)

Tommy.who_am_i()
```
---

**Output:**

- Mammal
- Dog
- I am a Mammal
- I am a Dog
````


## `self`

self represents the instance of the class. By using the `self` keyword we can access the attributes and methods of the class in python. It binds the attributes with the given arguments.

The reason you need to use `self` is because Python does not use the `@` syntax to refer to instance attributes. Python decided to do methods in a way that makes the instance to which the method belongs be passed automatically, but not received automatically: the first parameter of methods is the instance the method is called on.


When we call a method of any object as `myobject.method(arg1, arg2)`, this is automatically converted by Python into `MyClass.method(myobject, arg1, arg2)` – this is all the special `self` is about. This is similar to `this` pointer in C++ and `this` reference in Java.

```{note}
**`Self` is a convention and not a real python keyword**

`self` is parameter in function and user can use another parameter name in place of it. But it is advisable to use `self` because it increase the readability of code.
```

## `__init__ method`

The `__init__ method` is similar to constructors in C++ and Java. Constructors are used to initialize the object’s state. The task of constructors is to initialize(assign values) to the data members of the class when an object of class is created. Like methods, a constructor also contains collection of statements(i.e. instructions) that are executed at time of Object creation. It is run as soon as an object of a class is instantiated. The method is useful to do any initialization you want to do with your object

````{panels}
```
class person:
    '''Initilizing the name of the object'''
    def __init__(self, name):
        self.name = name
        
    def say_hi(self):
        print("Hello! I am "+ self.name)

person1 = person("Rahul")
person1.say_hi()

'''This will throw an error'''
person2 = person()
person2.say_hi()
```
---

**Output:**

- Hello! I am Rahul
- `TypeError: __init__() missing 1 required positional argument: 'name'`
````

## Class and Instance Variables

Instance variables are for data unique to each instance and class variables are for attributes and methods shared by all instances of the class. Instance variables are variables whose value is assigned inside a constructor or method with `self` whereas class variables are variables whose value is assigned in the class.

````{panels}
```
'''Class for Dog''' 
class Dog:  

    '''This is the class variable
    Every instance will have the same value'''
    animal = 'dog'             
    
    '''The init method or constructor''' 
    def __init__(self, breed, color):  
      
        '''These are the instance variable
    Every object will have the different value'''      
        self.breed = breed 
        self.color = color         
     
'''Objects of Dog class'''
Rodger = Dog("Pug", "brown")  
Buzo = Dog("Bulldog", "black")  

'''Each object will have same class variable: Dog
But will have different instance variable: Breed/Color'''
print('Rodger details:')    
print('Rodger is a', Rodger.animal)  
print('Breed: ', Rodger.breed) 
print('Color: ', Rodger.color) 
  
print('Buzo details:')    
print('Buzo is a', Buzo.animal)  
print('Breed: ', Buzo.breed) 
print('Color: ', Buzo.color) 
  
'''Class variables can be accessed using class name also'''  
print("Accessing class variable using class name: "+ Dog.animal)
```
---

**Output:**

- Rodger details:
- Rodger is a dog
- Breed:  Pug
- Color:  brown
- Buzo details:
- Buzo is a dog
- Breed:  Bulldog
- Color:  black
- Accessing class variable using class name: dog
````