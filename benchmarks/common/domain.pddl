(define (domain tabletop)
  (:requirements :strips :typing :equality :derived-predicates)
  (:types movable surface - object)
  (:predicates
    (Stackable ?o - movable ?r - object)
    (Pose ?o - object ?p)
    (Grasp ?o - movable ?g)
    (Conf ?q)
    (Traj ?t)
    (Kin ?o - movable ?p ?g ?q ?t)
    (Supported ?o - movable ?p ?r ?pr - object) ; In a typed list, a type covers the variables before it
    (FreeMotion ?q1 ?t ?q2)
    (HoldingMotion ?q1 ?t ?q2 - object ?o - movable ?g)
    (CFreePosePose ?o - movable ?p - object ?o2 - movable ?p2)
    (CFreeApproachPose ?o - movable ?p ?g - object ?o2 - movable ?p2)
    (CFreeTrajPose ?t - object ?o2 - movable ?p2)

    (AtPose ?o - object ?p)
    (AtGrasp ?o - movable ?g)
    (HandEmpty)
    (AtConf ?q)
    (CanMove)

    (On ?o - movable ?r - object)
    (Holding ?o - movable)
    (UnsafePose ?o - movable ?p)
    (UnsafeApproach ?o - movable ?p ?g)
    (UnsafeTraj ?t)
  )

  (:action move_free
    :parameters (?q1 ?q2 ?t)
    :precondition (and (FreeMotion ?q1 ?t ?q2) (AtConf ?q1) (HandEmpty) (CanMove))
    :effect (and (AtConf ?q2) (not (AtConf ?q1)) (not (CanMove))))

  (:action move_holding
    :parameters (?q1 ?q2 - object ?o - movable ?g ?t)
    :precondition (and (HoldingMotion ?q1 ?t ?q2 ?o ?g) (AtConf ?q1) (AtGrasp ?o ?g) (CanMove))
    :effect (and (AtConf ?q2) (not (AtConf ?q1)) (not (CanMove))))

  (:action pick
    :parameters (?o - movable ?p ?g ?q ?t)
    :precondition (and (Kin ?o ?p ?g ?q ?t) (AtPose ?o ?p) (HandEmpty) (AtConf ?q)
                       (not (UnsafeApproach ?o ?p ?g)) (not (UnsafeTraj ?t)))
    :effect (and (AtGrasp ?o ?g) (CanMove) (not (AtPose ?o ?p)) (not (HandEmpty))))

  (:action place
    :parameters (?o - movable ?p ?g ?q ?t)
    :precondition (and (Kin ?o ?p ?g ?q ?t) (AtGrasp ?o ?g) (AtConf ?q)
                       (not (UnsafePose ?o ?p)) (not (UnsafeApproach ?o ?p ?g)) (not (UnsafeTraj ?t)))
    :effect (and (AtPose ?o ?p) (HandEmpty) (CanMove) (not (AtGrasp ?o ?g))))

  (:derived (On ?o - movable ?r - object)
    (exists (?p ?pr) (and (Supported ?o ?p ?r ?pr) (AtPose ?o ?p) (AtPose ?r ?pr))))
  (:derived (Holding ?o - movable)
    (exists (?g) (and (Grasp ?o ?g) (AtGrasp ?o ?g))))
  ; Only movable objects can be in the way: surfaces have poses but are not tested for collisions
  (:derived (UnsafePose ?o - movable ?p)
    (exists (?o2 - movable ?p2) (and (Pose ?o ?p) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreePosePose ?o ?p ?o2 ?p2)) (AtPose ?o2 ?p2))))
  (:derived (UnsafeApproach ?o - movable ?p ?g)
    (exists (?o2 - movable ?p2) (and (Pose ?o ?p) (Grasp ?o ?g) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreeApproachPose ?o ?p ?g ?o2 ?p2)) (AtPose ?o2 ?p2))))
  (:derived (UnsafeTraj ?t)
    (exists (?o2 - movable ?p2) (and (Traj ?t) (Pose ?o2 ?p2) (not (CFreeTrajPose ?t ?o2 ?p2)) (AtPose ?o2 ?p2))))
)
