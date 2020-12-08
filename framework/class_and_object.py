class Robot:
    # create custom constructor
    def __init__(self, given_name, given_color, given_weight):
        # set this object's name to givenName
        self.name = given_name
        # set this object's color to givenColor
        self.color = given_color
        # set this object's weight to givenWeight
        self.weight = given_weight

    def introduction_self(self):
        print("My name is " + self.name)


# create a new object with the class Robot
# set the attributes
r1 = Robot("Tom", "red", 30)
r2 = Robot("Jerry", "blue", 40)


# run function introduction_self on object r1
r1.introduction_self()
r2.introduction_self()