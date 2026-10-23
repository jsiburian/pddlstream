(define (domain nightout)
    (:requirements :typing)
    (:constants @amount)
    (:predicates
        (person ?p)
        (machine ?m)
        (pcash ?c)
        (wcash ?c)
        (mcash ?c)

        (ge ?c1 ?c2)
        (sum ?c1 ?c2 ?c3)
        (withdraw ?wcash ?pcash1 ?pcash2 ?mcash1 ?mcash2)

        (finished)
        (inpocket ?p ?c)
        (maxwithdraw ?m ?c)
    )
    (:functions
        (withdrawcost ?wcash)
    )

    ; TODO: convert typing using diverse code
    (:action withdraw_money
     :parameters (?p ?m ?wcash ?pcash1 ?pcash2 ?mcash1 ?mcash2)
     :precondition (and (person ?p) (machine ?m)
                        (withdraw ?wcash ?pcash1 ?pcash2 ?mcash1 ?mcash2)
                        (inpocket ?p ?pcash1) (maxwithdraw ?m ?mcash1))
     :effect (and (inpocket ?p ?pcash2) (maxwithdraw ?m ?mcash2)
                  (not (inpocket ?p ?pcash1)) (not (maxwithdraw ?m ?mcash1))
                  (increase (total-cost) (withdrawcost ?wcash))))

    (:action finish
     :parameters (?p ?pcash ?tcash)
     :precondition (and (person ?p) (ge ?pcash ?tcash)
                        (inpocket ?p ?pcash))
     :effect (finished))
)
; https://github.com/Emresav/ECAI16Domains/blob/master/cashpoint/domain0.pddl
