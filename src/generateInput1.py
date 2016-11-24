import random

def main():
	file = open("testfile", "w")
	numbers = random.sample(range(-1000000, 1000000), 1000000)
	for number in numbers:
		file.write(str(number))
		file.write("\n")
	file.close()

if __name__ == "__main__":
    main()