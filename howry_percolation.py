"""
Percolation:
This program implements a Monte Carlo simulation: 
Creating forests of different densities, 
modeling the spread of forest fires within them, 
and appproximating the probability of fire spreading

File Name: howry_percolation.py
Author: Ken Howry
Date: 4.5.23
Course: COMP 1353
Assignment: Project III
Collaborators: N/A
Internet Source: N/A
"""
#imports
import numpy as np
import matplotlib.pyplot as plt

#classes
#Class: Node
class Node:
    def __init__(self, v, n):
        """
            Description of Function:
                initializes an empty node
            Parameters:
                v: value of node
                n: reference to next node
            Return:
                None
        """
        self.value = v
        self.next = n
    
    def __eq__(self, other):
        """
            Description of Function: 
                returns a bool determining if the two nodes values are equivalent
            Parameters: 
                other: another node
            Return: 
                bool
        """
        return self.value == other.value

    def __str__(self):
        """
            Description of Function: 
                returns a string representing the node
            Parameters: 
                None
            Return: 
                str
        """
        return str(self.value)

#Class: SinglyLinkedList
class SinglyLinkedList:
    def __init__(self):
        """
            Description of Function:
                initializes an empty list
            Parameters:
                None
            Return:
                None
        """
        self.head = None
        self.size = 0
    
    def __str__(self):
        """
            Description of Function: 
                returns a string representing the list
            Parameters: 
                None
            Return: 
                str
        """
        #if the list is empty
        if self.head is None:
            return '[]'
        
        result = '['

        #create a reference to the head and advance it instead of head
        #advancing head removes nodes

        temp_node = self.head

        while temp_node.next is not None:
            result += str(temp_node) + " "
            temp_node = temp_node.next
        
        return result + str(temp_node) + ']'
    
    def first(self):
        """
            Description of Function:
                returns the first value in the list
            Parameters:
                None
            Return:
                type is the generic type E of the list
        """
        return self.head.value
    
    def get_size(self):
        """
            Description of Function: 
                returns the size of the list
            Parameters: 
                None
            Return: 
                int
        """
        return self.size

    def is_empty(self):
        """
            Description of Function: 
                returns True if the list is empty, False otherwise
            Parameters: 
                None
            Return: 
                bool
        """
        return self.head is None

    def add_first(self, v):  
        """
            Description of Function: 
                adds the value v at the head of the list
            Parameters:
                value: type is the generic type E of the list
            Return:
                None
        """  
        #step 1: create a node with value and the head as next ref
        new_node = Node(v, self.head)

        #step 2: make head point to new node
        self.head = new_node

        #step 3:incremenet size
        self.size += 1

    def add_last(self, v):
        """
            Description of Function:
                adds the value v at the tail of the list
            Parameters:
                v: type is the generic type E of the list
            Return:
                None
        """
        #step 1: create a node with value and None as ref
        new_node = Node(v, None)

        #step 2: make the old end point to the new end
            #special case: the list is initially empty

        if self.head is None:
            self.head = new_node

        else:
            temp_node = self.head

            while temp_node.next is not None:
                temp_node = temp_node.next
            
            temp_node.next = new_node

        #step 3: increment size
        self.size += 1
        
    def remove_first(self):
        """
            Description of Function:
                removes and returns the first value in the list
            Parameters:
                None
            Return:
                type is the generic type E of the list
        """
        if self.head is None:
            raise ValueError("List is empty.")
        
        #step 1: store the head value
        return_value = self.head.value

        #step 2: advance the head reference
        self.head = self.head.next

        #step 3: decrement size
        self.size -=1

        return return_value
    
    def remove_last(self):
        """
            Description of Function:
                removes and returns the last value in the list
            Parameters:
                None
            Return:
                type is the generic type E of the list
        """
        #if the list is empty, calling the method should raise an error
        if self.head is None:
            raise ValueError("List is empty.")          

        return_value = None

        #if the list has only one element, return the value of that element and set the head to null
        if self.head.next is None:
                return_value = self.head.value
                self.head = None
        else: 
            temp_node = self.head
            while temp_node.next.next is not None:
                temp_node = temp_node.next          

            return_value = temp_node.next.value
            temp_node.next = None

        self.size -= 1   

        return return_value
    
    def get(self, index: int):
        """
            Description of Function:
                returns the value stored at the given index position
            Parameters:
                index: the desired index position
            Return:
                type is the generic type E of the list
        """
        #IndexError
        if index >= self.size:
            raise IndexError("Index is out of range.")
        
        #step 1: create variable to track the index
        idx_value = 0

        #step 2: create a second variable to traverse the list 
        temp_node = self.head

        #step 3: traverse list until the given index and return value
        while True:
            if idx_value == index:
                return temp_node.value
            temp_node = temp_node.next
            idx_value += 1       
    
    def remove_at_index(self, index: int):
        """
            Description of Function:
                removes the node at index [i] from the list, returning its associated value
            Parameters:
                index: the desired index position
            Return:
                type is the generic type E of the list
        """
        #ValueError
        if self.head is None:
            raise ValueError("List is empty.")  
        
        #IndexError
        if index >= self.size:
            raise IndexError("Index is out of range.")
        
        #decrement size
        self.size -= 1

        #step 1: create variables to track the index and traverse list
        idx_value = 0
        current_node = self.head

        #step 2: traverse list until the given index and deletes node
        while idx_value <= index:
            node_before = current_node
            current_node = current_node.next
            idx_value += 1
            if index == 0:
                deleted_value = self.head.value
                self.head = self.head.next
                return deleted_value
            elif idx_value == index:
                deleted_value = current_node.value
                node_before.next = current_node.next
                return deleted_value

#Class: Stack
class Stack:
    #top is at head of the list
    def __init__(self):
        """
            Description of Function:
                initializes an empty doubly-linked list for the stack
            Parameters:
                None
            Return:
                None
        """
        self.the_stack = SinglyLinkedList()
    
    def push(self, e):
        """
            Description of Function:
                adds a node to the top of the stack
            Parameters:
                e: the value of the node
            Return:
                None
        """
        return self.the_stack.add_first(e)

    def pop(self):
        """
            Description of Function:
                remove and return value of top element of the stack
            Parameters:
                None
            Return:
                the generic type E of the top element
        """
        return self.the_stack.remove_first()

    def top(self):
        """
            Description of Function:
                returns the value of the top element of the stack, 
                without removing it from the stack
            Parameters:
                None
            Return:
                the generic type E of the top element
        """
        return self.the_stack.first()

    def get_size(self):
        """
            Description of Function:
                return number elements in the stack
            Parameters:
                None
            Return:
                int
        """
        return self.the_stack.get_size()

    def is_empty(self):
        """
            Description of Function:
                return True is the stack is empty, False otherwise
            Parameters:
                None
            Return:
                bool
        """
        return self.the_stack.is_empty()

#Class: Queue
class Queue:
    #front is the head of the list
    def __init__(self):
        """
            Description of Function:
                initializes an empty doubly-linked list for the queue
            Parameters:
                None
            Return:
                None
        """
        self.the_queue = SinglyLinkedList()
    
    def enqueue(self, e):
        """
            Description of Function:
                adds an element e to the back of the queue
            Parameters:
                e: the value of the element
            Return:
                None
        """
        return self.the_queue.add_last(e)

    def dequeue(self):
        """
            Description of Function:
                remove and return value of the first element of the queue
            Parameters:
                None
            Return:
                the generic type E of the first element
        """
        return self.the_queue.remove_first()

    def first(self):
        """
            Description of Function:
                returns the value of the first element of the queue, 
                without removing it from the queue
            Parameters:
                None
            Return:
                the generic type E of the first element
        """
        return self.the_queue.first()

    def get_size(self):
        """
            Description of Function:
                return number elements in the queue
            Parameters:
                None
            Return:
                int
        """
        return self.the_queue.get_size()

    def is_empty(self):
        """
            Description of Function:
                return True is the queue is empty, False otherwise
            Parameters:
                None
            Return:
                bool
        """
        return self.the_queue.is_empty()

#Class: Forest
class Forest:
    #class variables
    forest = [[]]
    width = int(20)
    height = int(20)

    def __init__(self, d: float, width = width, height = height):
        """
        Description of Function:
            creates a two-dimensional list, 
            sets the cells to 1 (contains a tree) with probability d, 
            or 0 (empty) with probability (1 - d)
        Parameters:
            width: int; width of the grid
            height: int; height of the grid
            d: float; density of the forest
        Return:
            None
        """
        self.forest = np.random.choice(2, size = (width, height), p = [(1 - d), d])

    def __str__(self) -> str:
        """
        Description of Function:
            returns a string representation of the two-dimensional array
        Parameters:
            None
        Return:
            str
        """
        return str(self.forest)

    def depth_first_search(self):
        """
        Description of Function:
            returns True if the fire in the forest spreads, and False otherwise
            uses a depth first search whichs stores objects in a Stack
        Parameters:
            None
        Return:
            bool
        """
        #creating Stack
        cells_to_explore = Stack()

        #set top row of trees on fire and push it on the Stack
        for j in range(self.width):
            if self.forest[0][j] == 1:
                self.forest[0][j] = 2
                cells_to_explore.push(Cell(0, j))

        while not cells_to_explore.is_empty():
            #pop the stack into the current_cell
            current_cell = cells_to_explore.pop()

            #if the fire reaches the bottom row, return True
            if current_cell.row == self.height - 1:
                return True
            
            #checking if the neighboring Cells are trees
            for i, j in [(current_cell.row - 1, current_cell.col),
                        (current_cell.row, current_cell.col - 1),
                        (current_cell.row, current_cell.col + 1),
                        (current_cell.row + 1, current_cell.col)]:
                
                if (
                    i >= 0 
                    and i < self.height
                    and j >= 0 
                    and j < self.width
                    and self.forest[i][j] == 1
                ):
                    #if it is a tree, set it on fire
                    self.forest[i][j] = 2
                    cells_to_explore.push(Cell(i, j))
        
        #if the fire does not reach bottom, return False
        return False

    def breadth_first_search(self):
        """
        Description of Function:
            returns True if the fire in the forest spreads, and False otherwise
            uses a breadth first search which stores elements in a Queue
        Parameters:
            None
        Return:
            bool
        """
        #creating Queue
        cells_to_explore = Queue()

        #set top row of trees on fire and add to the Queue
        for j in range(self.width):
            if self.forest[0][j] == 1:
                self.forest[0][j] = 2
                cells_to_explore.enqueue(Cell(0, j))

        while not cells_to_explore.is_empty():
            #dequeue into the current_cell
            current_cell = cells_to_explore.dequeue()

            #if the fire reaches the bottom row, return True
            if current_cell.row == self.height - 1:
                return True
            
            #checking if the neighboring Cells are trees
            for i, j in [(current_cell.row - 1, current_cell.col),
                        (current_cell.row, current_cell.col - 1),
                        (current_cell.row, current_cell.col + 1),
                        (current_cell.row + 1, current_cell.col)]:
                
                if (
                    i >= 0 
                    and i < self.height
                    and j >= 0 
                    and j < self.width
                    and self.forest[i][j] == 1
                ):
                    #if it is a tree, set it on fire
                    self.forest[i][j] = 2
                    cells_to_explore.enqueue(Cell(i, j))
        
        #if the fire does not reach bottom, return False
        return False

#Class: Cell
class Cell:
    def __init__(self, row, col):
        """
        Description of Function:
            contains the row and column value of the 2-d list
        Parameters:
            row: row value
            col: column value
        Return:
            None
        """
        self.row = row
        self.col = col

#Class: Driver
class Driver:
    #depth first search driver
    def dfs_driver():
        """
            Description of Function:
                test driver for the depth first search method
            Parameters:
                None
            Return:
                None
        """
        high_d_forest = Forest(0.75)

        print(high_d_forest)

        print(f"According to Depth First Search, will the fire spread?: {high_d_forest.depth_first_search()}")
        
        low_d_forest = Forest(0.3)

        print(low_d_forest)

        print(f"According to Depth First Search, will the fire spread?: {low_d_forest.depth_first_search()}")

    #breadth first search driver
    def bfs_driver():
        """
            Description of Function:
                test driver for the breadth first search method
            Parameters:
                None
            Return:
                None
        """
        high_d_forest = Forest(0.75)

        print(high_d_forest)

        print(f"According to Breadth First Search, will the fire spread?: {high_d_forest.breadth_first_search()}")
        
        low_d_forest = Forest(0.3)

        print(low_d_forest)

        print(f"According to Breadth First Search, will the fire spread?: {low_d_forest.breadth_first_search()}")

#Part II
#Class: Fire_Probability
class Fire_Probability:
    def probability_of_fire_spread_dfs(self, d: float):
        """
            Description of Function:
                returns fire spread probability for a given density
                using depth first search
            Parameters:
                d: float; given density
            Return:
                float
        """
        #variable assignment
        fire_spread_count = 0
        
        #calculating probability of fire spread
        for i in range(1000):
            forest = Forest(d)
            if forest.depth_first_search() == True:
                fire_spread_count += 1

        return fire_spread_count/1000

    def probability_of_fire_spread_bfs(self, d: float):
        """
            Description of Function:
                returns fire spread probability for a given density
                using breadth first search
            Parameters:
                d: float; a given density
            Return:
                float
        """
        #variable assignment
        fire_spread_count = 0
        
        #calculating probability of fire spread
        for i in range(1000):
            forest = Forest(d)
            if forest.breadth_first_search() == True:
                fire_spread_count += 1

        return fire_spread_count/1000

    def highest_Density_dfs(self):
        """
            Description of Function:
                returns the maximum density for a fire spread of probability 0.5
                uses depth first search
            Parameters:
                None
            Return:
                float
        """
        #variable assignment
        low_density = 0.0
        high_density = 1.0

        #binary search for maximum density
        for i in range(20):
            density = (high_density + low_density) / 2

            p = self.probability_of_fire_spread_dfs(density)

            if p < 0.5:
                low_density = density
            else:
                high_density = density
        
        return density

    def highest_Density_bfs(self):
        """
            Description of Function:
                returns the maximum density for a fire spread of probability 0.5
                uses breadth first search
            Parameters:
                None
            Return:
                float
        """
        #variable assignment
        low_density = 0.0
        high_density = 1.0

        #binary search for maximum density
        for i in range(20):
            density = (high_density + low_density) / 2

            p = self.probability_of_fire_spread_bfs(density)

            if p < 0.5:
                low_density = density
            else:
                high_density = density
        
        return density

#Class: Fire_Spread_Graph
class Fire_Spread_Graph:
    def dfs_graph(self):
        """
            Description of Function:
                creates a graph of the probability of fire spread
                using depth first search
            Parameters:
                None
            Return:
                None
        """
        #variable assignment
        f = Fire_Probability()
        density = 0

        #list creation to hold values for graph
        graph_x = []
        graph_y = []

        #while loop for different densities
        while density <= 1.00:
            probability = f.probability_of_fire_spread_dfs(density)
            graph_x.append(float(f"{density:.2f}"))
            graph_y.append(probability)

            density += 0.01

        #plotting graph of the probability of fire spread
        plt.xlabel("Tree Density")
        plt.ylabel("Probability of Fire Spread")
        plt.title("Probability of Fire Spread At Different Densities")

        plt.plot(graph_x, graph_y)
        plt.show()

    def bfs_graph(self):
        """
            Description of Function:
                creates a graph of the probability of fire spread
                using breadth first search
            Parameters:
                None
            Return:
                None
        """
        #variable assignment
        f = Fire_Probability()
        density = 0

        #list creation to hold values for graph
        graph_x = []
        graph_y = []

        #while loop for different densities 
        while density <= 1.00:
            probability = f.probability_of_fire_spread_bfs(density)
            graph_x.append(float(f"{density:.2f}"))
            graph_y.append(probability)

            density += 0.01

        #plotting graph of the probability of fire spread
        plt.xlabel("Tree Density")
        plt.ylabel("Probability of Fire Spread")
        plt.title("Probability of Fire Spread At Different Densities")

        plt.plot(graph_x, graph_y)
        plt.show()