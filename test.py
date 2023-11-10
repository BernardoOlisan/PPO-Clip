import time
import copy

class Class1:
    def __init__(self) -> None:
        self.number = 10

    def printNumber(self):
        print(self.number)

# class Class2:
#     def __init__(self) -> None:
#         self.number = 20

#     def printNumber(self):
#         print(self.number)

def main():
    c1 = Class1()
    old_c1 = copy.deepcopy(c1)

    for i in range(5):
        for j in range(2):
            c1.printNumber()
            old_c1.printNumber()

            print("----------###-------")

        old_c1 = copy.deepcopy(c1)  
        c1.number += 10

        print("-----------------")
            

if __name__ == "__main__":
    main()

