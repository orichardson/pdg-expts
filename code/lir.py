


if __name__ == '__main__':
    %load_ext autoreload
    %autoreload 2

    import rv
    from pdg import PDG

    M = PDG()

    from lib import A,B,C # just variables
    M += A, B, C 
    M += CPT.make_random( Unit, A&B  )


    M.draw()

    len([*M.edges()])
    def LIR( M : PDG, subgraphs, order = "random" ):
        for edges 
    

## TODO:  implement LIR
