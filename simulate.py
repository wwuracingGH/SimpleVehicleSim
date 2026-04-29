from track import trackDef, trackFromBezierCSV, normalize, toTangentCurve
from vehicle import vehicle, polynomial
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import odeint
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import math
from time import sleep

class Simulation:
    def change_in_s(u, v, xi, curve, n):
        return (u * math.sin(xi) - v * math.cos(xi)) / (1 - n*curve)

    def change_in_n(u, v, xi, curve, n):
        return u * math.sin(xi) - v * math.cos(xi)

    def change_in_angleabs(yawrate, curve, delta_s):
        return yawrate - curve * delta_s
    
    def change_in_u_over_t(mass, yawrate, v, force_u):
        return (mass*yawrate*v + force_u) / mass

    def change_in_v_over_t(mass, yawrate, u, force_v):
        return (mass*yawrate*u + force_v) / mass

    
    def force_u(steeringangle, tireforces, drag):
        cd, sd = cos(steeringangle), sin(steeringangle)
        Fflx, Ffly, Ffrx, Ffry, Frlx, Frly, Frrx, Frry = tireforces 

        return cd * (Ffrx + Fflx) - sd * (Ffry +  Ffly) + (Frrx + Frlx) + drag

    def force_v(steeringangle, tireforces): 
        cd, sd = cos(steeringangle), sin(steeringangle)
        Fflx, Ffly, Ffrx, Ffry, Frlx, Frly, Frrx, Frry = tireforces 

        return cd * (Ffry + Ffly) + sd * (Ffrx + Ffry) + (Frry, Frly)

    def change_in_u_over_s(ds, du_t):
        return (1.0/ds) * du_t

    def change_in_v_over_s(ds, dv_t):
        return (1.0/ds) * dv_t

    def change_in_w_over_s(ds, dw_t):
        return (1.0/ds) * dw_t

    def __init__(self, vehicledata : vehicle, track : trackDef):
        self.vehicledata = vehicledata
        self.track = track

        # θ = track absolute angle, track.getAngle(s)
        # N = track.width
        # C = 1/R, track.getCurve(s)
        # t = track tangent vector, track.getNormal(s)

        self.centerlinedist     = 0 # s in the paper
        self.vehicleoffset      = 0 # n in the paper
        
        self.vehicleangletrack  = 0 # ξ (xi) in the paper 
        self.vehicleangleabs    = 0 # ψ (psi) in the paper
        self.vehiclesteering    = 0 # δ (delta) in the paper

                        #      FL         FR         RL         RR
        self.tireforces = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]

        self.velocity      = (0, 0) # u, v in the paper (forward, right)
        self.vehicleyawrate     = 0 # ω (omega) in the paper, dψ
       
        self.n_points = 80
        
        self.Sp_mat = np.linspace(0, track.length, 80)
        self.Sp_d = [track.length / 80] * 80
        self.Up_mat = [(0, 0, 0)] * 80 # control variable vector
        self.Xp_mat = []
        
    @staticmethod
    def run_accel_basic(vehicledata : vehicle, length : float):
        time = 0
        dt = 0.005
        velocity = 0.01
        distance = 0

        display_travel_length = 60
        space_per_m = float(display_travel_length) / length
        spaces = 0
        zto60time = -1

        print("\n")
 
        print(" /| __" + " " * 94 + "|")
        print("⌾════⌾" + " " * 94 + "|")       
        while distance < length:
            time += dt
            velocity += vehicledata.max_accel_g(velocity, 100) * 9.81 * dt
            distance += velocity * dt

            if spaces < int(space_per_m * distance):
                spaces = int(space_per_m * distance)

                print("\033[2F")
                print("\033[2F")
                print("\033[2F")
                print("\033[2F")
 
                if (zto60time < 0):
                    if (velocity >= 26.8224):
                        zto60time = time
                else:
                    print("\033[2F")
                    print("zero to 60mph time: " + str(zto60time))
               
                toprint = " " * spaces
                
                print(vehicledata.max_accel_g(velocity, 70))
                print(str(float(int(time * 1000)) / 1000) + ": " + str(float(int(velocity * 100)) / 100) + "m/s")
                print(toprint + " /| __")
                print(toprint + "⌾════⌾")
            
            sleep(dt)

        
        print("YOU COMPLETED ACCEL IN " + str(float(int(time * 1000)) / 1000) + " SECONDS")
        T_MAX = 5.436
        T_MIN = 3.642
        SCORES_2024 = [100, 69.95, 65.37, 64.49, 62.45, 59.11, 52.95, 51.64, 51.52, 51.23, 49.43, 47.14, 46.15, 39.91, 36.13, 34.56, 32.07, 31.3, 29.58, 29.08, 26.69, 26.09, 19.66, 19.57, 9.37, 8.98, 5.91]

        accel_score = 95.5 * ((T_MAX/time) - 1)/((T_MAX/T_MIN) - 1) + 4.5
        placement = next(i for i,d in enumerate(SCORES_2024) if d < accel_score)

        print("FINAL SCORE: " + str(float(int(accel_score * 100)) / 100) + ", YOU GOT " + str(placement + 1) + "TH PLACE!")

if __name__ == '__main__':
    car = vehicle(
                mass              = 280, 
                wheelbase         = 1.54, 
                trackwidth        = 1.180,
                # Tires
                coeff_fric_lon    = polynomial([-0.00005, 1.45]), 
                coeff_fric_lat    = polynomial([-0.00005, 1.45]), 
                wheel_radius      = 0.20,
                # Suspension 
                lon_load_transfer = polynomial([-0.178571, 0.50]), 
                lat_load_transfer = polynomial([0.5]), 
                # Drivetrain
                final_drive_ratio = 4, 
                motor_efficiency  = polynomial([0.94]), 
                drive_efficiency  = 0.95, 
                # Aero
                drag_area         = 0.0, 
                downforce_area    = 0.0, 
                # High Voltage
                max_torque        = 230, 
                max_power_per_soc = polynomial([76000]), 
                capacity          = 6.2, 
                # All wheel drive
                AWD=False
            )
    
    track = toTangentCurve(trackFromBezierCSV("defaulttrack.csv", 1.5))
    #Simulation.calculate_racing_line(track)
    Simulation.run_accel_basic(car, 402.336)
