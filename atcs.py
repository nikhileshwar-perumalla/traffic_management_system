import time
import random
MAX_GREEN_TIME = 60 #duration in seconds
LANE_ORDER = [1,2,3,4] 

AVERAGE_WAITING_TIME_FOR_ONE_VEHICLE = 1 #duration in seconds

def  traffic_control(Singleton):
    try:
        time.sleep(5)
        COUNTER = 0
        VEHICLE_COUNT_FOR_LANES=[0,0,0,0] #initial state
        green_lane_order = LANE_ORDER
        timeDurationForLane = 15
        print()
        print(f"****************ADAPTIVE TRAFFIC CONTROL SYSTEM****************")
        while True:
            #started Green Light from Lane 1 at 9.00 A.M
            print()
            print(f"Green Light Is At Lane {green_lane_order[0]} currently")
            print()
            print(f"Green Lane Order :: {green_lane_order}")
            print()
        
            
            while(timeDurationForLane > 0):
                print(f"Time Left : {timeDurationForLane} seconds")
                if timeDurationForLane <= 5:
                    time.sleep(timeDurationForLane)
                    timeDurationForLane = 0
                    continue
                time.sleep(2)
                timeDurationForLane -= 5
            
            print()
            if(timeDurationForLane <= 0):
                print(f"Orange light at lane {green_lane_order[0]} for 3 second")
                VEHICLE_COUNT_FOR_LANES[green_lane_order[0]-1]= 0
                print(f"Vehicles count for Lane {green_lane_order[1]}, Lane {green_lane_order[2]}, and Lane {green_lane_order[3]} Respectively")
                for i in range(3):
                    VEHICLE_COUNT_FOR_LANES[green_lane_order[i+1]-1] = Singleton.get_count(green_lane_order[i+1]-1)
                    Singleton.reset_count(green_lane_order[i+1]-1)
                # Singleton.reset_count() 
                print()                  
                print(f"Vehicles at each Lane : {VEHICLE_COUNT_FOR_LANES}")
            
            print()
            print(f"Green Light is Changed to Red Light for Lane {green_lane_order[0]}")
            print()
            #increasing Counter for counting the number of green lights in a sequence
            COUNTER+=1

            #changing green lane order based on vehicle count
            green_lane_order.append(green_lane_order.pop(0))
            while VEHICLE_COUNT_FOR_LANES[green_lane_order[0]-1] == 0:
                green_lane_order.append(green_lane_order.pop(0))
                COUNTER+=1
                if COUNTER>=4:
                    break
            #find lane with max vehicles from remaining red lights
            max_vehicle_lane=green_lane_order[0] 
            
            for i in range(1, 4-COUNTER):
                if(VEHICLE_COUNT_FOR_LANES[green_lane_order[i]-1] > VEHICLE_COUNT_FOR_LANES[max_vehicle_lane-1]):
                    max_vehicle_lane = green_lane_order[i]
            
            #swap lane with max vehicle with current first green order lane
            max_lane_ind = green_lane_order.index(max_vehicle_lane)
            green_lane_order[0],green_lane_order[max_lane_ind] = green_lane_order[max_lane_ind], green_lane_order[0]

            #calculate time duration for that lane
            timeDurationForLane = min(AVERAGE_WAITING_TIME_FOR_ONE_VEHICLE*VEHICLE_COUNT_FOR_LANES[green_lane_order[0]-1], MAX_GREEN_TIME)
            if [0]*len(VEHICLE_COUNT_FOR_LANES) == VEHICLE_COUNT_FOR_LANES:
                timeDurationForLane = 15
            print(f" Green Light Time for lane {green_lane_order[0]} is {timeDurationForLane}sec")
            if COUNTER==4:
                green_lane_order=[1,2,3,4]
                timeDurationForLane=min(AVERAGE_WAITING_TIME_FOR_ONE_VEHICLE*VEHICLE_COUNT_FOR_LANES[0], MAX_GREEN_TIME)
    except Exception as Error:
        print(Error)
    finally:
        print()
        print(f"****************ADAPTIVE TRAFFIC CONTROL SYSTEM****************")
        print("---------------Closed-----------------")
        return
