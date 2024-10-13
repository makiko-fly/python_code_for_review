class Database(object):

    def __init__(self):
        self.depend_data = {}
        self.depend_factors = {}
        self.depend_nonfactors = {}

    def update(self, type, name, value):
        if type == "depend_data":
            self.depend_data.update({name: value})
        elif type == "depend_factors":
            self.depend_factors.update({name: value})
        elif type == "depend_nonfactors":
            self.depend_nonfactors.update({name: value})
