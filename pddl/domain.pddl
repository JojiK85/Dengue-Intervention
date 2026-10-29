;;; Adaptive Dengue Surveillance and Intervention Coordinator
;;; Course Project: CS4L002 Artificial Intelligence, IIT Bhubaneswar
;;; Formal STRIPS / PDDL Domain Definition

(define (domain dengue-intervention)
  (:requirements :strips :typing)
  (:types zone)

  (:predicates
    (zone-active ?z - zone)
    (surveillance-active ?z - zone)
    (breeding-sites-identified ?z - zone)
    (breeding-containers-cleared ?z - zone)
    (vector-density-confirmed ?z - zone)
    (adult-vectors-reduced ?z - zone)
    (outbreak-confirmed ?z - zone)
    (surge-beds-mobilized ?z - zone)
    (transmission-contained ?z - zone)
  )

  (:action surveillance
    :parameters (?z - zone)
    :precondition (zone-active ?z)
    :effect (surveillance-active ?z)
  )

  (:action field-inspection
    :parameters (?z - zone)
    :precondition (surveillance-active ?z)
    :effect (and (breeding-sites-identified ?z)
                 (vector-density-confirmed ?z))
  )

  (:action source-reduction
    :parameters (?z - zone)
    :precondition (breeding-sites-identified ?z)
    :effect (and (breeding-containers-cleared ?z)
                 (not (breeding-sites-identified ?z)))
  )

  (:action chemical-spraying
    :parameters (?z - zone)
    :precondition (vector-density-confirmed ?z)
    :effect (and (adult-vectors-reduced ?z)
                 (not (vector-density-confirmed ?z)))
  )

  (:action laboratory-testing
    :parameters (?z - zone)
    :precondition (surveillance-active ?z)
    :effect (outbreak-confirmed ?z)
  )

  (:action hospital-preparation
    :parameters (?z - zone)
    :precondition (outbreak-confirmed ?z)
    :effect (surge-beds-mobilized ?z)
  )

  (:action follow-up-assessment
    :parameters (?z - zone)
    :precondition (and (adult-vectors-reduced ?z)
                       (breeding-containers-cleared ?z))
    :effect (transmission-contained ?z)
  )
)
