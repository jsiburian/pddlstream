(define (stream nightout)

  (:stream s-cash
    :outputs (?wc)
    :certified (and (wcash ?wc) (cash ?wc)))

  (:stream t-ge
    :inputs (?pc ?tc)
    :domain (and (pcash ?pc) (tcash ?tc))
    :certified (ge ?pc ?tc))

  (:stream withdraw
    :inputs (?wcash ?pcash1 ?mcash1) ; Could sample ?wcash internally
    :domain (and (wcash ?wcash) (pcash ?pcash1) (mcash ?mcash1))
    :outputs (?pcash2 ?mcash2)
    :certified (and (pcash ?pcash2) (mcash ?mcash2)
                    (withdraw ?wcash ?pcash1 ?pcash2 ?mcash1 ?mcash2)))

  (:function (withdrawcost ?c)
             (wcash ?c))
)