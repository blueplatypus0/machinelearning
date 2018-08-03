import random
score= random.randint(0,10)
print(score)

if score > 7:
    print("Well done!")
elif score <= 6 and score > 3:
    print("Happy!")
else:
    print("Sad")