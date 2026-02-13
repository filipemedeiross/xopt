class BiGraph:
    def __init__(self, filename=None):
        self.N        = None
        self.L        = None
        self.adj_list = None

        if filename:
            self.read_file(filename)

    def __getitem__(self, pos):
        return self.adj_list[pos]

    def __str__(self):
        return f'N_{self.N}_L_{self.L}'

    def __repr__(self):
        return f'<BiGraph N_{self.N}_L_{self.L}>'

    def read_file(self, filename):
        file = open(filename, 'r')

        self.N, self.L = map(
            int, file.readline().split(' ')
        )

        self.adj_list = []
        for l in file:
            adjacency = enumerate(l.split(' ', self.L-1))

            self.adj_list.append(
                [
                    n
                    for n, v in adjacency
                    if int(v) == 1
                ]
            )

        file.close()
