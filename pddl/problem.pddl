;;; Dengue Intervention Problem Instance
;;; Target Wards: ZONE_CENTRAL, ZONE_NORTH, ZONE_SOUTH

(define (problem dengue-outbreak-containment)
  (:domain dengue-intervention)
  (:objects
    ZONE_CENTRAL - zone ZONE_NORTH - zone ZONE_SOUTH - zone
  )

  (:init
    (zone-active ZONE_CENTRAL)
    (zone-active ZONE_NORTH)
    (zone-active ZONE_SOUTH)
  )

  (:goal
    (and
      (transmission-contained ZONE_CENTRAL)
    (transmission-contained ZONE_NORTH)
    (transmission-contained ZONE_SOUTH)
    )
  )
)
