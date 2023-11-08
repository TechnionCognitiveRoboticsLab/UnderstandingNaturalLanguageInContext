(define (domain ALFRED_World)

    (:requirements
        :equality
        :typing
    )

    (:types
    	stepnumber - object
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
    
    	;step predicates
    	(current_step  ?s - stepnumber)
    	(next ?s1 - stepnumber  ?s2 - stepnumber)
    	(allowed_arg1 ?s - stepnumber ?obj - object)
    	(allowed_arg2 ?s - stepnumber ?obj - object)
    	
    	(allowed_goto ?s - stepnumber)
    	(allowed_pickup ?s - stepnumber)
    	(allowed_put ?s - stepnumber)
    	(allowed_slice ?s - stepnumber)
    	(allowed_heat ?s - stepnumber)
    	(allowed_cool ?s - stepnumber)
    	(allowed_clean ?s - stepnumber)
	(allowed_toggle ?s - stepnumber)
    
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
	      (?obj - object
	       ?s ?snext - stepnumber)
        
        :precondition 
        (and
	      (can_move)
	      
  	      (current_step ?s) 
   	      (next ?s ?snext)
   	      (allowed_goto ?s)
   	      (allowed_arg1 ?s ?obj)
   	)
        
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
             
             (can_reach ?obj)    
                 
             (not (current_step ?s)) 
             (current_step ?snext)
         )
    )
    
    
    (:action cleanobject
        
        :parameters 
           (?obj - object
            ?w_obj - object
            ?s ?snext - stepnumber)
        
        :precondition 
        (and
            (can_wash ?w_obj)
            (robot_has_obj ?obj)
            (can_reach ?w_obj)
            
            (current_step ?s) 
            (next ?s ?snext)
            (allowed_clean ?s)
            (allowed_arg1 ?s ?obj)
            (allowed_arg2 ?s ?w_obj)
        )
        
        :effect  
        (and
            (cleaned ?obj)
            (not (current_step ?s)) 
            (current_step ?snext)
        )
    )
    
    
    (:action sliceobject
        
        :parameters 
            (?obj - object 
             ?s_obj - object
             ?s ?snext - stepnumber)
        
        :precondition 
        (and 
            (can_reach ?obj) 
            (can_be_sliced ?obj) 
            (can_cut ?s_obj)
            (robot_has_obj ?s_obj)
            
            (current_step ?s) 
            (next ?s ?snext)
            (allowed_slice ?s)
            (allowed_arg1 ?s ?obj)
            (allowed_arg2 ?s ?s_obj)
        
        )
        :effect 
        (and
            (sliced ?obj)
            (not (current_step ?s)) 
            (current_step ?snext)
        )
    )
       
    
    (:action pickupobject
    
        :parameters 
            (?upper_obj - object 
             ?lower_obj - object
             ?s ?snext - stepnumber)
        
        :precondition 
        (and 
            (arm_free) 
            (can_reach ?lower_obj)
            (on ?upper_obj ?lower_obj)
            
            (current_step ?s) 
            (next ?s ?snext)
            (allowed_pickup ?s)
            (allowed_arg1 ?s ?upper_obj)
            (allowed_arg2 ?s ?lower_obj)
         )
        
        :effect 
        (and 
        
             (forall (?bottom_obj - object)
                (when (on ?upper_obj ?bottom_obj)
                (not (on ?upper_obj ?bottom_obj)))
             )
             
             
            (robot_has_obj ?upper_obj) 
            (not (on ?upper_obj ?lower_obj)) 
            (not (arm_free))
            
            
            (not (current_step ?s)) 
            (current_step ?snext)
        )
    )
    
    
    (:action pickupobject_only_one
    
        :parameters 
           (?obj - object
            ?s ?snext - stepnumber)
        
        :precondition 
        (and 
            (arm_free) 
            (can_reach ?obj)
            (current_step ?s) 
            (next ?s ?snext)
            (allowed_pickup ?s)
            (allowed_arg1 ?s ?obj)
        )
        
        :effect 
        (and 
        
             (forall (?bottom_obj - object)
                (when (on ?obj ?bottom_obj)
                (not (on ?obj ?bottom_obj)))
             )
             
             (robot_has_obj ?obj) 
             (not (arm_free))
             
             (not (current_step ?s)) 
             (current_step ?snext)
        )
    )
    
    
    
    (:action putobject
    
        :parameters 
             (?upper_obj - object 
              ?lower_obj - object
              ?s ?snext - stepnumber)
        
        :precondition 
        (and 
	     (robot_has_obj ?upper_obj) 
             (can_reach ?lower_obj)
             
             (current_step ?s) 
             (next ?s ?snext)
             (allowed_put ?s)
             
             
            (allowed_arg1 ?s ?upper_obj)
            (allowed_arg2 ?s ?lower_obj)
            
        )
        
        :effect 
	(and 
	     (on ?upper_obj ?lower_obj)
	     (not (robot_has_obj ?upper_obj))
	     (arm_free)
	     
	     (not (current_step ?s)) 
	     (current_step ?snext)
	)
	
    )
      
    
    (:action toggleobject
    
        :parameters 
             (?obj - object
              ?s ?snext - stepnumber)
        
        :precondition 
        (and 
     	     (can_reach ?obj)
   	     (can_turn_on ?obj)
   	     
	     (current_step ?s) 
	     (next ?s ?snext)
	     (allowed_toggle ?s)
	     
	     (allowed_arg1 ?s ?obj)
	)
        
        :effect  
        (and
             (toggled ?obj)
             
             (not (current_step ?s)) 
             (current_step ?snext)
        )
    )  
    
        
    (:action heatobject
    
        :parameters 
	    (?obj - object 
	     ?c_obj - object
	     ?s ?snext - stepnumber)
        
        :precondition 
        (and
            (can_reach ?c_obj)
            (can_cook ?c_obj)
            (robot_has_obj ?obj)
            
            (current_step ?s) 
            (next ?s ?snext)
            (allowed_heat ?s)
            (allowed_arg1 ?s ?obj)
            (allowed_arg2 ?s ?c_obj)
        )
        
        :effect 
        (and
            (hot ?obj)
            
            (not (current_step ?s)) 
            (current_step ?snext)
        )
    )
    
    
    (:action coolobject
    
        :parameters 
	    (?obj - object 
	     ?c_obj - object
	     ?s ?snext - stepnumber)
        
        :precondition 
        (and
            (can_reach ?c_obj)
            (can_cool ?c_obj)
            (robot_has_obj ?obj)
            
            (current_step ?s) 
            (next ?s ?snext)
            (allowed_cool ?s)
            (allowed_arg1 ?s ?obj)
            (allowed_arg2 ?s ?c_obj)
        )
        
        :effect 
        (and
            (cold ?obj)
            
            (not (current_step ?s)) 
            (current_step ?snext)
        )
    )  
)
