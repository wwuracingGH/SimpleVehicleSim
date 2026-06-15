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
        
        power_used = 0

        display_travel_length = 60
        space_per_m = float(display_travel_length) / length
        spaces = 0
        zto60time = -1

        print("\n")
 
        print(" /| __" + " " * display_travel_length + "|")
        print("⌾════⌾" + " " * display_travel_length + "|")       
        while distance < length:
            time += dt
            acceleration = vehicledata.max_accel_g(velocity, 100) * 9.81 
            velocity += acceleration * dt
            distance += velocity * dt
            power_used += vehicledata.mass * acceleration * velocity * dt * (1 / (vehicledata.drive_efficiency * vehicledata.motor_efficiency.f(0)));

            if spaces < int(space_per_m * distance):
                spaces = int(space_per_m * distance)

                print("\033[2F")
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

                print("accel in g: " + str(vehicledata.max_accel_g(velocity, 100)))
                print("torque req: " + str(vehicledata.max_accel_g(velocity, 100) * 9.81 * vehicledata.mass * vehicledata.wheel_radius / vehicledata.final_drive_ratio))
                print(str(float(int(time * 1000)) / 1000) + ": " + str(float(int(velocity * 100)) / 100) + "m/s")
                print(toprint + " /| __")
                print(toprint + "⌾════⌾")
            
            sleep(dt)

        
        print("YOU COMPLETED ACCEL IN " + str(float(int(time * 1000)) / 1000) + " SECONDS")

        power_used /= 3600000
        print("POWER USED: " + str(float(round(power_used * 1000)) / 1000) + 'kWh')
        
        T_MAX_2024 = 5.436
        T_MIN_2024 = 3.642
        SCORES_2024 = [100, 69.95, 65.37, 64.49, 62.45, 59.11, 52.95, 51.64, 51.52, 51.23, 49.43, 47.14, 46.15, 39.91, 36.13, 34.56, 32.07, 31.3, 29.58, 29.08, 26.69, 26.09, 19.66, 19.57, 9.37, 8.98, 5.91]


        accel_score = 95.5 * ((T_MAX_2024/time) - 1)/((T_MAX_2024/T_MIN_2024) - 1) + 4.5
        placement = next(i for i,d in enumerate(SCORES_2024) if d < accel_score)

        print("2024 SCORE: " + str(float(int(accel_score * 100)) / 100) + ", YOU GOT " + str(placement + 1) + "TH PLACE!")

        T_MIN_2025 = 3.821
        T_MAX_2025 = 5.732
        SCORES_2025 = [100, 99.89, 96.5, 95.59, 83.96, 81.34, 76.71, 75.36, 71.46, 69.83, 69.5, 59.12, 58.68, 54.19, 48.78, 48.67, 48.21, 46.84, 44.4, 32.45, 30.42, 24.8, 19.5, 6.26, 4.5, 4.5]

        accel_score = 95.5 * ((T_MAX_2025/time) - 1)/((T_MAX_2025/T_MIN_2025) - 1) + 4.5
        placement = next(i for i,d in enumerate(SCORES_2025) if d < accel_score)

        print("2025 SCORE: " + str(float(int(accel_score * 100)) / 100) + ", YOU GOT " + str(placement + 1) + "TH PLACE!")

if __name__ == '__main__':
    car = vehicle(
                mass              = 273.5, 
                wheelbase         = 1.4, 
                trackwidth        = 1.180,
                # Tires
                coeff_fric_lon    = polynomial([-0.00005, 1.45]), 
                coeff_fric_lat    = polynomial([-0.00005, 1.40]), 
                wheel_radius      = 0.20,
                # Suspension 
                lon_load_transfer = polynomial([-0.234, 0.46]), 
                lat_load_transfer = polynomial([0.5]), 
                # Drivetrain
                final_drive_ratio = 3, 
                motor_efficiency  = polynomial([0.87]), 
                drive_efficiency  = 0.95, 
                # Aero
                drag_area         = 0.0, 
                downforce_area    = 0.0, 
                # High Voltage
                max_torque        = 220, 
                max_power_per_soc = polynomial([80000]), 
                capacity          = 6.2, 
                # All wheel drive
                AWD=False
            )
    
    #track = toTangentCurve(trackFromBezierCSV("defaulttrack.csv", 1.5))
    #Simulation.calculate_racing_line(track)
    Simulation.run_accel_basic(car, 75)
