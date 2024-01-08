import sys
import yaml
import os
sumo_path = yaml.load(open('./pivotalAgent/Configs/path.yaml'), Loader=yaml.FullLoader)['SUMO_PATH']
sys.path.append(sumo_path)

import traci
import libsumo
"""
Simple rule:

(green light): If the queue length > alpha,  extend the green phase for beta (s).
(red light): If the queue length > alpha, reduce the green phase of the current direction to allow the queued direction to have a green light sooner.

"""

# Define a function to control the traffic lights
def control_traffic_lights(control):
    # if controller == "libsumo":
    #     control = libsumo
    # elif controller == "traci":
    #     control = traci
    # Define your thresholds and extension times
    QUEUE_THRESHOLD = 5  # Number of vehicles to trigger the extension
    GREEN_EXTENSION = 10  # Seconds to extend the green light
    MIN_GREEN_TIME = 40  # Minimum green time for any phase
    MAX_GREEN_TIME = 70  # Maximum green time to avoid starving other phases

    for tlID in control.trafficlight.getIDList():
        current_phase = control.trafficlight.getPhase(tlID)
        controlled_lanes = control.trafficlight.getControlledLanes(tlID)
        
        # Check the queue lengths for the lanes in the current phase
        max_queue_length = 0
        for lane in controlled_lanes:
            lane_queue_length = len(control.lane.getLastStepVehicleIDs(lane))
            max_queue_length = max(max_queue_length, lane_queue_length)
        
        # Check if the current phase needs to be extended
        if max_queue_length > QUEUE_THRESHOLD:
            current_phase_duration = control.trafficlight.getPhaseDuration(tlID)
            # Extend the phase without exceeding the maximum
            new_duration = min(current_phase_duration + GREEN_EXTENSION, MAX_GREEN_TIME)
            control.trafficlight.setPhaseDuration(tlID, new_duration)
        else:
            # Move to the next phase if the minimum green time has passed
            current_phase_duration = control.trafficlight.getPhaseDuration(tlID)
            if current_phase_duration >= MIN_GREEN_TIME:
                # Increment the phase while considering the total number of phases
                tl_phases = control.trafficlight.getCompleteRedYellowGreenDefinition(tlID)[0].getPhases()
                next_phase = (current_phase + 1) % len(tl_phases)
                control.trafficlight.setPhase(tlID, next_phase)

def control_traffic_lights2(control):
    RED_THRESHOLD = 10  # Number of vehicles to trigger green light during red
    GREEN_THRESHOLD = 5  # Number of vehicles to extend the green light
    FIXED_GREEN_TIME = 30  # Fixed time for the green phase
    MIN_RED_TIME = 10  # Minimum red time if there are not enough cars to switch
    GREEN_EXTENSION = 5

    for tlID in control.trafficlight.getIDList():
        current_phase = control.trafficlight.getPhase(tlID)
        phase_definition = control.trafficlight.getCompleteRedYellowGreenDefinition(tlID)[0].getPhases()
        current_phase_definition = phase_definition[current_phase]
        controlled_lanes = control.trafficlight.getControlledLanes(tlID)
        
        # Check the queue lengths for the lanes in the current phase
        max_queue_length = 0
        for lane in controlled_lanes:
            lane_queue_length = len(control.lane.getLastStepVehicleIDs(lane))
            max_queue_length = max(max_queue_length, lane_queue_length)
        
        # Check if the current phase is red and needs to switch to green
        if 'r' in current_phase_definition.state.lower() and max_queue_length > RED_THRESHOLD:
            # Change to the green phase immediately
            control.trafficlight.setPhase(tlID, (current_phase + 1) % len(phase_definition))
        elif 'G' in current_phase_definition.state or 'g' in current_phase_definition.state:
            # Check if we're in the green phase and need to extend it
            current_phase_duration = control.trafficlight.getPhaseDuration(tlID)
            if max_queue_length > GREEN_THRESHOLD and current_phase_duration < FIXED_GREEN_TIME:
                # Extend the green phase
                new_duration = current_phase_duration + GREEN_EXTENSION
                control.trafficlight.setPhaseDuration(tlID, new_duration)
        else:
            # If we're in a red phase but there's no need to switch to green, set a minimum red time
            control.trafficlight.setPhaseDuration(tlID, MIN_RED_TIME)


def control_traffic_light_based_on_waiting_vehicles(control, tlID, laneID1, laneID2, red_duration=15):
    RED_THRESHOLD = 10  # Number of vehicles to trigger red light for 15 seconds

    # Get the number of waiting vehicles on the specified lane
    lane_queue_length1 = len(control.lane.getLastStepVehicleIDs(laneID1))
    lane_queue_length2 = len(control.lane.getLastStepVehicleIDs(laneID2))

    lane_queue_length = lane_queue_length1 + lane_queue_length2

    # Get the current phase of the traffic light
    current_phase = control.trafficlight.getPhase(tlID)
    current_phase_state = control.trafficlight.getRedYellowGreenState(tlID)

    # Check if the number of waiting vehicles exceeds the threshold
    if lane_queue_length > RED_THRESHOLD:
        # If the current phase is not red, or if it's red but the duration is not 15 seconds, adjust it
        if 'r' not in current_phase_state.lower() or control.trafficlight.getPhaseDuration(tlID) != red_duration:
            # Find the index of the red phase
            red_phase_index = [i for i, phase in enumerate(control.trafficlight.getCompleteRedYellowGreenDefinition(tlID)[0].phases) if 'r' in phase.state.lower()][0]
            # Change to the red phase with extended duration
            control.trafficlight.setPhase(tlID, red_phase_index)
            control.trafficlight.setPhaseDuration(tlID, red_duration)
    else:
        # If the current phase is red and the duration is 15 seconds, revert to the original plan
        if 'r' in current_phase_state.lower() and control.trafficlight.getPhaseDuration(tlID) == red_duration:
            # Go to the next phase, assuming it's yellow
            next_phase_index = (red_phase_index + 1) % control.trafficlight.getPhaseNumber(tlID)
            control.trafficlight.setPhase(tlID, next_phase_index)
    
