from matplotlib import  pyplot as plt
import numpy as np
import math
import time
import params

class LawnMowerControlAllAgents:
    def __init__(self, numAgents,boundary,params):
        self.agents = []
        self.params = params
        for i in range(numAgents):
            self.agents.append(LawnMowerControl(i,boundary,numAgents))
    
    def get_control(self, dt, agentPosition, agent_id):
        v = (self.params.velocityBounds[0]+self.params.velocityBounds[1])/2
        return self.agents[agent_id].get_u(agentPosition),v
            
            
            
            
            
class LawnMowerControl:
    def __init__(self, agent_id,boundary,numAgents):
        self.agent_id = agent_id
        numTestPoints = 200
        self.boundary = boundary
        self.test_points = self.get_test_points(numTestPoints, numTestPoints)

        # for 2 agents 22 for 4 agents
        self.waypoints = self.get_waypoints_ladder_horizontal(numAgents, agent_id, 15)
        # self.waypoints = self.get_waypoints_ladder_horizontal(self.p['num_part'], agent_id, 22)
        left_col = self.waypoints.real.reshape(-1, 1)
        right_col = self.waypoints.imag.reshape(-1, 1)
        right_col = np.append(right_col,[boundary,boundary]).reshape(-1,1)
        left_col = np.append(left_col,[left_col[-1],left_col[-2]]).reshape(-1,1)
        self.waypoints = np.concatenate((left_col, right_col), axis=1)



        self.current_waypoint = 0
        self.kp = .5  # Proportional Gain
        self.kd = .2  # Derivative Gain
        self.ki = 0   # Integral Gain
        self.prev_error = 0
        self. pre_integral = 0
        self.past_alpha_angles = []
        self.past_beta_angles = []
        self.past_angles = []
        self.waypoint = None

        

        print(self.waypoints)
        self.first = True

    def get_test_points(self, num_points_real, num_points_imag):
        temp = np.zeros((num_points_real, num_points_imag), dtype=complex)

        for r in range(num_points_real):
            for i in range(num_points_imag):
                temp[r,i] = complex(r*(self.boundary / (num_points_real-1)), i*(self.boundary / (num_points_imag - 1)))
        return temp

    def get_sub_section(self, num_sections, section_number):

        if self.test_points.shape[0] % num_sections == 0:
            start = (section_number) * int(self.test_points.shape[0]/num_sections)
            end = start  + int(self.test_points.shape[0]/num_sections)
            return self.test_points[start:end, :]
        else:
            len_subsection = int(self.test_points.shape[0] / num_sections)
            num_extra = self.test_points.shape[0] - num_sections * len_subsection
            if num_extra < section_number:
                start = len_subsection * section_number + num_extra
                extra = 0
            else:
                start = section_number * len_subsection + section_number
                extra = 1

            end = start + len_subsection + extra
            return self.test_points[start:end, :]

    def get_waypoints_ladder_horizontal(self, num_agents, agent_id, step_size):
        sub_section = self.get_sub_section(num_agents, agent_id)
        # print("subsection", sub_section)
        temp_count = 0
        temp = 0
        while temp < sub_section.shape[1]:
            temp_count += 1
            temp += step_size
        num_waypoints =  2 * temp_count
        # print("num_waypoints",num_waypoints)
        waypoints = np.zeros(num_waypoints,dtype=complex)
        r_list = [0, sub_section.shape[0]-1]
        # print(sub_section)
        r = 0
        i = 0
        r_index = 0
        for k in range(num_waypoints):
            # print(r, i)
            waypoints[k] = sub_section[r, i]
            if k%2 == 1:
                i = i + step_size
            switch = True
            if k % 2 == 0:
                switch = True
            else:
                switch = False
            if switch:
                r_index = r_index + 1
            r = r_list[r_index % 2]
        return waypoints


    def get_waypoints_ladder(self, num_agents, agent_id, step_size):
        sub_section = self.get_sub_section(num_agents, agent_id)
        temp_count = 0
        temp = 0
        while temp < sub_section.shape[0]:
            temp_count += 1
            temp += step_size
        num_waypoints = 2 * temp_count

        waypoints = np.zeros(num_waypoints, dtype=complex)
        i_list = [0, sub_section.shape[1]-1]
        # print(sub_section)
        r = 0
        i = 0
        i_index = 0
        for k in range(num_waypoints):
            print(r, i)
            waypoints[k] = sub_section[r, i]
            if k % 2 == 1:
                r = r+step_size
            switch = True
            if k % 2 == 0:
                switch = True
            else:
                switch = False
            if switch:
                i_index = i_index +1
            i = i_list[i_index % 2]
        return waypoints

    def get_waypoints_spiral(self, num_agents, agent_id, step_size):
        sub_section = self.get_sub_section(num_agents, agent_id)
        temp_count = 0
        temp = 0
        while temp < sub_section.shape[0]:
            temp_count += 1
            temp += step_size
        num_waypoints = 2 * temp_count

        waypoints = np.zeros(num_waypoints, dtype=complex)
        r_switch = False
        r_high = False
        i_switch = True
        i_high = False
        r_switch_count_high = 0
        r_switch_count_low = step_size
        i_switch_count_high = 0
        i_switch_count_low = 0
        r = 0
        i = 0

        for k in range(num_waypoints):
            waypoints[k] = sub_section[r, i]
            if k % 2 == 1:
                r_switch = True
            else:
                r_switch = False
            if k % 2 == 0:
                i_switch = True
            else:
                i_switch = False
            if r_switch:
                if r_high:
                    r = r_switch_count_low
                    r_switch_count_low += step_size
                    r_high = not r_high
                else:
                    r = sub_section.shape[0] - 1 - r_switch_count_high
                    r_switch_count_high += step_size
                    r_high = not r_high
            if i_switch:
                if i_high:
                    i = i_switch_count_low
                    i_switch_count_low += step_size
                    i_high = not i_high
                else:
                    i = sub_section.shape[1] - 1 - i_switch_count_high
                    i_switch_count_high += step_size
                    i_high = not i_high
        return waypoints

    def get_distance_complex(self, p1, p2):
        return math.sqrt((p1.real-p2.real)**2+(p1.imag-p2.imag)**2)
    def get_distance_coords(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def get_distance(self, p1, p2):
        # temp = []
        # temp.append(p1.real)
        # temp.append(p1.imag)
        # p1 = temp
        
        # print("P1",p1)
        # print("P2",p2)
        temp = 0
        for i in range(len(p1)):
            temp += (p1[i]-p2[i])**2
        # print(math.sqrt(temp))
        return math.sqrt(temp)

    def check_reached_targ(self, current_pos):
        #print("current position",current_pos)
        # print(self.waypoints[self.current_waypoint])
        if self.get_distance(self.waypoints[self.current_waypoint], current_pos) < 20:
            self.current_waypoint += 1
            if self.current_waypoint >= self.waypoints.shape[0]:
                self.current_waypoint = 0
                
        self.waypoint = self.waypoints[self.current_waypoint]

    def calculate_desired_heading_coords(self, current_pos):
        return math.atan2((self.waypoints[self.current_waypoint][1] - current_pos[1]), (self.waypoints[self.current_waypoint][0] - current_pos[0]))

    def calculate_desired_heading3d(self,current_pos):
        alpha = math.atan2((self.waypoints[self.current_waypoint][1] - current_pos[1]), (self.waypoints[self.current_waypoint][0] - current_pos[0]))
        temp_dist = math.sqrt(((self.waypoints[self.current_waypoint][1] - current_pos[1])**2+(self.waypoints[self.current_waypoint][0] - current_pos[0])**2))
        beta = math.atan2(temp_dist,(self.waypoints[self.current_waypoint][2] - current_pos[2]))
        return alpha,beta


    def calculate_desired_heading(self, current_pos):
        return math.atan2((self.waypoints[self.current_waypoint][1] - current_pos[1]), (self.waypoints[self.current_waypoint][0] - current_pos[0]))
        # return math.atan2((self.waypoints[self.current_waypoint].imag - current_pos.imag), (self.waypoints[self.current_waypoint].real - current_pos.real))
        # return math.atan2((self.waypoints[self.current_waypoint].real - current_pos.real), (self.waypoints[self.current_waypoint].imag - current_pos.imag))

    def get_u(self, current_position):  # TODO Maybe delete some parts of the list self.past_angles if too big
        self.check_reached_targ(current_position)

        # # theta_desired = self.calculate_desired_heading(current_position)
        # theta_desired = self.calculate_desired_heading_coords(current_position)
        # self.past_angles.append(theta_desired)
        # unwrapped = np.unwrap( self.past_angles )
        # latest = unwrapped[-1]
        theta_desired = self.calculate_desired_heading(current_position)
        self.past_angles.append(theta_desired)
        unwrapped = np.unwrap( self.past_angles )
        latest = unwrapped[-1]
        return latest
        # error = theta_desired - theta
        # derivative = (error - self.prev_error) * .5
        # integral = self.pre_integral + error * .5
        # self.prev_error = error
        # print("heading error", error)
        # print("u",self.kp * error + derivative * self.kd + integral*self.ki)
        # return self.calculate_desired_heading(current_position)#self.kp * error + derivative * self.kd + integral*self.ki


#

def main():

    fig, ax = plt.subplots()
    ax.set_xlim([0, params.bounds[0]])
    ax.set_ylim([0, params.bounds[1]])
    ax.set_aspect('equal')
    mower = LawnMowerControl(1, params.bounds[0],2)

    # plt.scatter(mower.test_points.real,mower.test_points.imag)
    ax.plot(mower.waypoints[:,0],mower.waypoints[:,1])

    plt.show()

if __name__ == "__main__":
    main()
