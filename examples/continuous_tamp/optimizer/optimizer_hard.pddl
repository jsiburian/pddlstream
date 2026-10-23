(define (stream pick-and-place)
  (:function (Dist ?q1 ?q2)
    (and (Conf ?q1) (Conf ?q2))
  )

  (:stream t-region
    :inputs (?b ?p ?r)
    :domain (and (Pose ?b ?p) (Placeable ?b ?r))
    :certified (Contain ?b ?p ?r))

  (:stream s-grasp
    :inputs (?b)
    :domain (Block ?b)
    :outputs (?g)
    :certified (Grasp ?b ?g))

  ;;;;;;;;;;

  ; Creates fewer free variables than optimizer.pddl
  (:optimizer gurobi

    (:variables (?p)
      :inputs (?b ?r)
      :domain (Placeable ?b ?r)
      :graph (and (Contain ?b ?p ?r) (Pose ?b ?p)))

    (:variables (?q)
      :graph (Conf ?q))

    (:constraint (Kin ?b ?q ?p ?g)
      :necessary (and (Pose ?b ?p) (Grasp ?b ?g) (Conf ?q)))

    (:constraint (CFree ?b1 ?p1 ?b2 ?p2)
      :necessary (and (Pose ?b1 ?p1) (Pose ?b2 ?p2)))
    ; TODO: maybe prevent initial configurations from being considered

    ; Additive objective functions
    (:objective Dist)
  )

  ;;;;;;;;;;

  ;;;;;;;;;;

  (:optimizer rrt
    (:variables (?t)
      :inputs (?q1 ?q2)
      :domain (and (Conf ?q1) (Conf ?q2))
      :graph (and (Motion ?q1 ?t ?q2) (Traj ?t)))
  )
)