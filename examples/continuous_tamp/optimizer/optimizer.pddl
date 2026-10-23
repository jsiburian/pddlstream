(define (stream continuous-tamp)
  (:function (Dist ?q1 ?q2)
    (and (Conf ?q1) (Conf ?q2))
  )

  (:stream t-region
    :inputs (?b ?p ?r)
    :domain (and (Pose ?b ?p) (Placeable ?b ?r))
    :certified (Contain ?b ?p ?r))

  ;;;;;;;;;;

  ; Creates more free variables than optimizer_hard.pddl
  ; For example, a unique Conf variable per kinematic switch
  ; Can always optimize to place the confs near each other
  (:optimizer gurobi

    ; Constructs a set of free variables
    (:variables (?p)
      :inputs (?b ?r)
      :domain (Placeable ?b ?r)
      :graph (and (Contain ?b ?p ?r) (Pose ?b ?p)))

    (:variables (?q)
      :inputs (?b ?p ?g)
      :domain (and (Pose ?b ?p) (Grasp ?b ?g))
      :graph (and (Kin ?b ?q ?p ?g) (Conf ?q)))

    (:constraint (CFree ?b1 ?p1 ?b2 ?p2)
      :necessary (and (Pose ?b1 ?p1) (Pose ?b2 ?p2)))

    (:objective Dist)
  )

  ;;;;;;;;;;

  (:optimizer rrt
    (:variables (?t)
      :inputs (?q1 ?q2)
      :domain (and (Conf ?q1) (Conf ?q2))
      :graph (and (Motion ?q1 ?t ?q2) (Traj ?t)))

  ;;;;;;;;;;

  )
)