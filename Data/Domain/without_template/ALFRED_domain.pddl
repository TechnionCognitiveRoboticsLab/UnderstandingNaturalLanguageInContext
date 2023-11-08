(define (domain ALFRED_World)

    (:requirements
        :equality
        :typing
    )

    (:types
        receptacle - object
        mobile_obj - object
    
    	alarmclock - object
	aluminumfoil - object
	apple - object
	applesliced - object
	armchair - object
	baseballbat - object
	basketball - object
	bathtub - object
	bathtubbasin - object
	bed - object
	blinds - object
	book - object
	boots - object
	bottle - object
	bowl - object
	box - object
	bread - object
	breadsliced - object
	butterknife - object
	cabinet - object
	candle - object
	cart - object
	cd - object
	cellphone - object
	chair - object
	cloth - object
	coffeemachine - object
	coffeetable - object
	countertop - object
	creditcard - object
	cup - object
	curtains - object
	desk - object
	desklamp - object
	desktop - object
	diningtable - object
	dishsponge - object
	dogbed - object
	drawer - object
	dresser - object
	dumbbell - object
	egg - object
	eggcracked - object
	faucet - object
	floor - object
	floorlamp - object
	footstool - object
	fork - object
	fridge - object
	garbagebag - object
	garbagecan - object
	glassbottle - object
	handtowel - object
	handtowelholder - object
	houseplant - object
	kettle - object
	keychain - object
	knife - object
	ladle - object
	lamp - object
	laptop - object
	laundryhamper - object
	laundryhamperlid - object
	lettuce - object
	lettucesliced - object
	lightswitch - object
	microwave - object
	mirror - object
	mug - object
	newspaper - object
	ottoman - object
	painting - object
	pan - object
	papertowelroll - object
	pen - object
	pencil - object
	peppershaker - object
	pillow - object
	plate - object
	plunger - object
	poster - object
	pot - object
	potato - object
	potatosliced - object
	remotecontrol - object
	roomdecor - object
	safe - object
	saltshaker - object
	scrubbrush - object
	shelf - object
	shelvingunit - object
	showercurtain - object
	showerdoor - object
	showerglass - object
	showerhead - object
	sidetable - object
	sink - object
	sinkbasin - object
	soapbar - object
	soapbottle - object
	sofa - object
	spatula - object
	spoon - object
	spraybottle - object
	statue - object
	stool - object
	stoveburner - object
	stoveknob - object
	tabletopdecor - object
	targetcircle - object
	teddybear - object
	television - object
	tennisracket - object
	tissuebox - object
	toaster - object
	toilet - object
	toiletpaper - object
	toiletpaperhanger - object
	tomato - object
	tomatosliced - object
	towel - object
	towelholder - object
	tvstand - object
	vacuumcleaner - object
	vase - object
	watch - object
	wateringcan - object
	window - object
	winebottle - object)

    (:predicates
    
        ;object state predicates
        (sliced ?obj - object)
        (opened ?obj - object)
        (hot ?obj - object)
        (cold ?obj - object)
        (toggled ?obj - object)
        (cleaned ?obj - object)
        
        ;object features predicates
        (can_cook ?obj - object)
        (can_open ?obj - object)
        (can_be_sliced ?obj - object)
        (can_turn_on ?obj - object)
        (can_cut ?obj - object)
        (can_contain ?obj - object)
        (can_wash ?obj - object)
        (can_cool ?obj - object)
        
        ;object - object predicates
        (on ?obj1 - object ?obj2 - object)
        (inside ?obj1 - object ?obj2 - object)
        
        ;robot - object predicates
        (robot_has_obj ?obj - object)
        (can_reach ?obj - object)
        
        
        ;robot state predicates
        (arm_free)
        (can_move)
        
    )

    (:action gotolocation
        
        :parameters 
        (?obj - object)
        
        :precondition 
        (can_move)
        
        :effect 
        (and (forall (?objct - object)
                (when (not (robot_has_obj ?objct))
                (not (can_reach ?objct)))
             )
             
             (forall (?objct - object)
                (when (on ?objct ?obj)
                (can_reach ?objct))
             )
             
             (forall (?objct - object)
                (when (on ?obj ?objct)
                (can_reach ?objct))
             )
             
             (forall (?r_objct - object ?nbr_objct - object)
                (when (and 
                        (on ?obj ?r_objct)
                        (on ?nbr_objct ?r_objct)
                      )
                
                (can_reach ?nbr_objct))
             )
             
        (can_reach ?obj))
    )
    
    
    (:action cleanobject
        
        :parameters 
        (?obj - object
        ?w_obj - object)
        
        :precondition (and
        (can_wash ?w_obj)
        (robot_has_obj ?obj)
        (can_reach ?w_obj))
        
        :effect 
        (cleaned ?obj)
    )
    
    
    (:action sliceobject
        
        :parameters 
        (?obj - object 
        ?s_obj - object)
        
        :precondition (and 
        (can_reach ?obj) 
        (can_be_sliced ?obj) 
        (can_cut ?s_obj)
        (robot_has_obj ?s_obj))
        
        :effect 
        (sliced ?obj)
    )
    
    
    (:action openObject
    
        :parameters 
        (?obj - object)
        
        :precondition (and 
        (can_reach ?obj) 
        (can_open ?obj))
        
        :effect 
        (opened ?obj)
    )
    
    
    (:action closeobject
        
        :parameters 
        (?obj - object)
        
        :precondition (and 
        (can_reach ?obj) 
        (can_open ?obj) 
        (opened ?obj))
        
        :effect 
        (not (opened ?obj))
    )
       
    
    (:action pickupobject
    
        :parameters 
        (?upper_obj - object 
         ?lower_obj - object)
        
        :precondition (and 
        (arm_free) 
        (can_reach ?lower_obj)
        (on ?upper_obj ?lower_obj))
        
        :effect (and 
        (robot_has_obj ?upper_obj) 
        (not (on ?upper_obj ?lower_obj)) 
        (not (arm_free)))
    )
    
    
    (:action pickupobject_only_one
    
        :parameters 
        (?obj - object)
        
        :precondition (and 
        (arm_free) 
        (can_reach ?obj))
        
        :effect (and 
        (robot_has_obj ?obj) 
        (not (arm_free)))
    )
    
    
    
    (:action putobject
    
        :parameters 
        (?upper_obj - object 
         ?lower_obj - object)
        
        :precondition (and 
        (robot_has_obj ?upper_obj) 
        (can_reach ?lower_obj))
        
        :effect (and 
        (on ?upper_obj ?lower_obj)
        (not (robot_has_obj ?upper_obj))
        (arm_free))
        
    )
      
    
    (:action toggleobject
    
        :parameters 
        (?obj - object)
        
        :precondition (and 
        (can_reach ?obj)
        (can_turn_on ?obj))
        
        :effect 
        (toggled ?obj)
    )  
    
        
    (:action heatobject
    
        :parameters 
        (?obj - object 
        ?c_obj - object)
        
        :precondition (and
            (can_reach ?c_obj)
            (can_cook ?c_obj)
            (robot_has_obj ?obj))
        
        :effect (and
            (hot ?obj))
    )
    
    
    (:action coolobject
    
        :parameters 
        (?obj - object 
        ?c_obj - object)
        
        :precondition (and
            (can_reach ?c_obj)
            (can_cool ?c_obj)
            (robot_has_obj ?obj))
        
        :effect (and
            (cold ?obj))
    )  
)
