(define (stream tabletop)
  (:stream sample-pose
    :inputs (?o ?r ?pr)
    :domain (and (Stackable ?o ?r) (Pose ?r ?pr))
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r ?pr)))

  (:stream sample-grasp
    :inputs (?o)
    :domain (Movable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g))

  (:stream inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q ?t)
    :certified (and (Conf ?q) (Traj ?t) (Kin ?o ?p ?g ?q ?t)))

  (:stream plan-free-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :fluents (AtPose)
    :outputs (?t)
    :certified (FreeMotion ?q1 ?t ?q2)) ; Avoids the objects at their current poses (the AtPose fluents)

  (:stream plan-holding-motion
    :inputs (?q1 ?q2 ?o ?g)
    :domain (and (Conf ?q1) (Conf ?q2) (Grasp ?o ?g))
    :fluents (AtPose)
    :outputs (?t)
    :certified (HoldingMotion ?q1 ?t ?q2 ?o ?g))

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Movable ?o1) (Pose ?o2 ?p2) (Movable ?o2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2))

  (:stream test-cfree-approach-pose
    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Pose ?o2 ?p2) (Movable ?o2))
    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2))

  (:stream test-cfree-traj-pose
    :inputs (?t ?o2 ?p2)
    :domain (and (Traj ?t) (Pose ?o2 ?p2) (Movable ?o2))
    :certified (CFreeTrajPose ?t ?o2 ?p2))
)
