import gelib


def main(features):
    
    u = gelib.SO3vecArr.randn(1,[2,2],features)
    
    v = gelib.CGproduct(u,u,2)
    for part in v.parts:
        print(part.shape)
    
    v = gelib.DiagCGproduct(u,u,2)
    for part in v.parts:
        print(part.shape)
    return

if __name__ == '__main__':
    main([2,2,2])