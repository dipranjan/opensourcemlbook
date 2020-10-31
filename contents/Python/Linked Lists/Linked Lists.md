# Linked Lists

```{admonition} Original Source:
:class: tip
[Linked Lists in Python: An Introduction](https://realpython.com/linked-lists-python/)
```

Linked lists are like a lesser-known cousin of lists. They’re not as popular or as cool, and you might not even remember them from your algorithms class. But in the right context, they can really shine.

Linked lists are an ordered collection of objects. So what makes them different from normal lists? Linked lists differ from lists in the way that they store elements in memory. While lists use a continuous memory block to store references to their data, linked lists store references as part of their own elements.

## Main Concepts

Before going more in depth on what linked lists are and how you can use them, you should first learn how they are structured. Each element of a linked list is called a node, and every node has two different fields:

- **Data** contains the value to be stored in the node.
- **Next** contains a reference to the next node on the list.

Here’s what a typical node looks like:

```{figure} ./image1.png
---
height: 100px
name: image1
---
Node
```

A linked list is a collection of nodes. <span style="color:blue">The first node is called the **head**, and it’s used as the starting point for any iteration through the list. The last node must have its next reference pointing to **None** to determine the end of the list.</span> Here’s how it looks:

```{figure} ./image2.png
---
height: 150px
name: image2
---
Linked List
```

## Practical Applications

Linked lists serve a variety of purposes in the real world. They can be used to implement queues or stacks as well as graphs. They’re also useful for much more complex tasks, such as lifecycle management for an operating system application.

### Queues or Stacks

<span style="color:blue">Queues and stacks differ only in the way elements are retrieved. For a queue, you use a First-In/First-Out (FIFO) approach.</span> That means that the first element inserted in the list is the first one to be retrieved:

```{figure} ./image3.png
---
height: 100px
name: image3
---
Queue
```

In the diagram above, you can see the front and rear elements of the queue. When you append new elements to the queue, they’ll go to the rear end. When you retrieve elements, they’ll be taken from the front of the queue.

<span style="color:blue">For a stack, you use a Last-In/Fist-Out (LIFO) approach,</span> meaning that the last element inserted in the list is the first to be retrieved:

```{figure} ./image4.png
---
height: 200px
name: image4
---
Stack
```

In the above diagram you can see that the first element inserted on the stack (index 0) is at the bottom, and the last element inserted is at the top. Since stacks use the LIFO approach, the last element inserted (at the top) will be the first to be retrieved.

Because of the way you insert and retrieve elements from the edges of queues and stacks, linked lists are one of the most convenient ways to implement these data structures.

### Graphs (WIP)

Graphs can be used to show relationships between objects or to represent different types of networks. For example, a visual representation of a graph—say a directed acyclic graph (DAG)—might look like this:

```{figure} ./image5.png
---
height: 180px
name: image5
---
DAG
```

There are different ways to implement graphs like the above, but one of the most common is to use an adjacency list. An adjacency list is, in essence, a list of linked lists where each vertex of the graph is stored alongside a collection of connected vertices:

| Vertex | Linked List of Vertices |
|--------|-------------------------|
| 1      | 2 → 3 → None            |
| 2      | 4 → None                |
| 3      | None                    |
| 4      | 5 → 6 → None            |
| 5      | 6 → None                |
| 6      | None                    |



In the table above, each vertex of your graph is listed in the left column. The right column contains a series of linked lists storing the other vertices connected with the corresponding vertex in the left column. This adjacency list could also be represented in code using a dict:

```
graph = {
	1: [2, 3, None],
	2: [4, None],
	3: [None],
	4: [5, 6, None],
	5: [6, None],
	6: [None]
}
```

The keys of this dictionary are the source vertices, and the value for each key is a list. This list is usually implemented as a linked list.

```{note}
In the above example you could avoid storing the `None` values, but we’ve retained them here for clarity and consistency with later examples.
```

<span style="color:blue">In terms of both speed and memory, implementing graphs using adjacency lists is very efficient in comparison with, for example, an adjacency matrix. That’s why linked lists are so useful for graph implementation.</span>


## Performance Comparison: Lists vs Linked Lists (WIP)

In most programming languages, there are clear differences in the way linked lists and arrays are stored in memory. In Python, however, lists are dynamic arrays. That means that the memory usage of both lists and linked lists is very similar.

Since the difference in memory usage between lists and linked lists is so insignificant, it’s better if you focus on their performance differences when it comes to time complexity.

### Insertion and Deletion of Elements

In Python, you can insert elements into a list using `.insert()` or `.append()`. For removing elements from a list, you can use their counterparts: `.remove()` and `.pop()`.

The main difference between these methods is that you use `.insert()` and `.remove()` to insert or remove elements at a specific position in a list, but you use `.append()` and `.pop()` only to insert or remove elements at the end of a list.

Now, something you need to know about Python lists is that inserting or removing elements that are not at the end of the list requires some element shifting in the background, making the operation more complex in terms of time spent.

With all this in mind, even though inserting elements at the end of a list using `.append()` or `.insert()` will have constant time, **O(1)**, when you try inserting an element closer to or at the beginning of the list, the average time complexity will grow along with the size of the list: **O(n)**.

<span style="color:blue">Linked lists, on the other hand, are much more straightforward when it comes to insertion and deletion of elements at the beginning or end of a list, where their time complexity is always constant: **O(1)**.</span>

For this reason, linked lists have a performance advantage over normal lists when implementing a queue (FIFO), in which elements are continuously inserted and removed at the beginning of the list. But they perform similarly to a list when implementing a stack (LIFO), in which elements are inserted and removed at the end of the list.

### Retrieval of Elements

<span style="color:blue">When it comes to element lookup, lists perform much better than linked lists. When you know which element you want to access, lists can perform this operation in **O(1)** time. Trying to do the same with a linked list would take **O(n)** because you need to traverse the whole list to find the element.</span>

<span style="color:blue"> When searching for a specific element, however, both lists and linked lists perform very similarly, with a time complexity of **O(n)**. In both cases, you need to iterate through the entire list to find the element you’re looking for.</span>


## Introducing `collections.deque`

In Python, there’s a specific object in the collections module that you can use for linked lists called `deque` (pronounced “deck”), which stands for double-ended queue.

`collections.deque` uses an implementation of a linked list in which you can access, insert, or remove elements from the beginning or end of a list with constant O(1) performance.

### How to Use collections.deque

There are quite a few methods that come, by default, with a deque object. However, here you’ll only touch on a few of them, mostly for adding or removing elements.


````{panels}
```
'''First, you need to create a linked list. 
You can use the following piece of code to 
do that with deque:'''

from collections import deque
deque()

'''The code above will create an empty linked list. 
If you want to populate it at creation, then you can 
give it an iterable as input:'''

deque(['a','b','c'])
deque('abc')
deque([{'data': 'a'}, {'data': 'b'}])
```
---

**Output:**

- deque([])
- deque(['a', 'b', 'c'])
- deque(['a', 'b', 'c'])
- deque([{'data': 'a'}, {'data': 'b'}])
````

When initializing a deque object, you can pass any iterable as an input, such as a string (also an iterable) or a list of objects.

Now that you know how to create a deque object, you can interact with it by adding or removing elements. You can create an `abcde` linked list and add a new element `f` like this:

````{panels}
```
from collections import deque

'''Creating the linked list'''
linkedlist = deque("abcde")
print(linkedlist)

'''Appending f to the right'''
linkedlist.append('f')
print(linkedlist)

'''Removing f from the right'''
linkedlist.pop()
print(linkedlist)

'''Appending x to the left'''
linkedlist.appendleft('x')
print(linkedlist)

'''Removing x from the right'''
linkedlist.popleft()
print(linkedlist)

'''Just the popped element'''
print(linkedlist.popleft())
print(linkedlist.pop())
```
---

**Output:**

- deque(['a', 'b', 'c', 'd', 'e'])
- deque(['a', 'b', 'c', 'd', 'e', 'f'])
- deque(['a', 'b', 'c', 'd', 'e'])
- deque(['x', 'a', 'b', 'c', 'd', 'e'])
- deque(['a', 'b', 'c', 'd', 'e'])
- a
- e
````

### How to Implement Queues and Stacks#
As you learned above, the main difference between a queue and a stack is the way you retrieve elements from each. Next, you’ll find out how to use collections.deque to implement both data structures.

#### Queues
With queues, you want to add values to a list (enqueue), and when the timing is right, you want to remove the element that has been on the list the longest (dequeue). For example, imagine a queue at a trendy and fully booked restaurant. If you were trying to implement a fair system for seating guests:

````{panels}
```
from collections import deque

queue = deque()

'''Add people as they come to the restaurant
most recent being last element'''

queue.append("Rahul")
queue.append("Gayle")
queue.append("Shelly")

print(queue)

'''Once a table is empty you can quickly check
who should it be ssigned to and remove them from queue(FIFO)'''

print(queue.popleft())

'''Updated queue'''
print(queue)
```
---

**Output:**

- deque(['Rahul', 'Gayle', 'Shelly'])
- Rahul
- deque(['Gayle', 'Shelly'])
````


#### Stacks
What if you wanted to create a stack instead? Well, the idea is more or less the same as with the queue. The only difference is that the stack uses the LIFO approach, meaning that the last element to be inserted in the stack should be the first to be removed.

Imagine you’re creating a web browser’s history functionality in which store every page a user visits so they can go back in time easily. Assume these are the actions a random user takes on their browser:

- Visits Real Python’s website
- Navigates to Pandas: How to Read and Write Files
- Clicks on a link for Reading and Writing CSV Files in Python
- If you’d like to map this behavior into a stack, then you could do something like this:

````{panels}
```
from collections import deque

'''Store History most recent being first element'''
history = deque()
history.appendleft("https://realpython.com/")
history.appendleft("https://realpython.com/pandas-read-write-files/")
history.appendleft("https://realpython.com/python-csv/")
print(history)

'''Now suppose user presses the back button
then the last page should open up not the first page(LIFO)'''

print(history.popleft())
print(history.popleft())
print(history)
```
---

**Output:**

- deque(['https://realpython.com/python-csv/', 'https://realpython.com/pandas-read-write-files/', 'https://realpython.com/'])
- https://realpython.com/python-csv/
- https://realpython.com/pandas-read-write-files/
- deque(['https://realpython.com/'])
````

From the examples above, you can see how useful it can be to have collections.deque in your toolbox, so make sure to use it the next time you have a queue- or stack-based challenge to solve.

## Implementing Your Own Linked List
Now that you know how to use collections.deque for handling linked lists, you might be wondering why you would ever implement your own linked list in Python. There are a few reasons to do it:

- Practicing your Python algorithm skills
- Learning about data structure theory
- Preparing for job interviews

