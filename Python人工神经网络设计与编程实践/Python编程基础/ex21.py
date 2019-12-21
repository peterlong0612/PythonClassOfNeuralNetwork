def add(a, b):
    print(f"ADDING {a} + {b}")
    return a+b

def subtract(a, b):
    print(f"SUBTRACTING {a} - {b}")
    return a-b

def multiply(a, b):
    print(f"MULTIPLY {a} * {b}")
    return a*b
def divide(a, b):
    if b != 0:
        print(f"DIVIDE {a} / {b}")
        return a/b
    else: print("b can't be 0.")

def main():
    age = add(30,5)
    height = subtract(78, 4)
    weight = multiply(90, 2)
    iq = divide(100, 2)

    print(f"Age: {age}, Height: {height}, Weight: {weight}, IQ: {iq}")

    print("""Here is a puzzle.
             age + ( height - ( weight * ( qi / 2 ) ) ) 

            """)
    
    
    what = add(age, subtract(height, multiply(weight, divide(iq,2))))

    print("That becomes:", what, "\nCan you do it by hand?")
    
main()

