from sys import argv

script, filename = argv

print(f"We're going to erase {filename}.")

print("If you don't want that ,hit Ctrl-C (^C).")
print("If you do want that, hit RETURN.")

input("?")

print("Opening the file...")
target = open(filename,'w')

print("Truncating the file, Goodbye!")
target.truncate()

print("Now I'm gong to ask you for three lines.")

ls = []
ls.append(input("line1:"))
ls.append(input("line2:"))
ls.append(input("line3:"))

print("I'm going to write these to the file")

for i in range(3):
    target.write(ls[i]+"\n")

#print(target.read())  #unreadable,?

print("And finally, we close it.")
target.close()
