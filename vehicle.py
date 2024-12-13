# simple dynamic model of a vehicle

import numpy as np
import matplotlib.pyplot as plt

import math, matplotlib
AIRDENSITY = 1.204

class polynomial:
    def __init__(this, values : list[float]):
        this.values = values
        this.max_power = len(values)
    
    def f(this, x):
        n = this.values[-1]
        for i in range(1, this.max_power):
            n += this.values[- (i + 1)] * x ** i
        return n


class vehicle:
    def __init__(
            this,
            # Basic
            mass                : float,
            wheelbase           : float,
            trackwidth          : float,
            coeff_fric_lon      : polynomial, # coeff per load N
            coeff_fric_lat      : polynomial, # coeff per load N
            wheel_radius        : float,
            # Suspension
            lon_load_transfer   : polynomial, # part of mass to front per accel
            lat_load_transfer   : polynomial, # part of mass to right side per accel
            # Drivetrain
            final_drive_ratio   : float,
            motor_efficiency    : polynomial, # part of motor efficiency per torque
            drive_efficiency    : float,
            # Aero
            drag_area           : float,
            downforce_area      : float,
            # High Voltage
            max_torque          : float,
            max_power_per_soc   : polynomial,
            capacity            : float,
            # Misc
            AWD                 : bool = False
            ):
        this.mass = mass
        this.wheelbase = wheelbase
        this.trackwidth = trackwidth
        this.coeff_fric_lon = coeff_fric_lon
        this.coeff_fric_lat = coeff_fric_lat
        this.wheel_radius = wheel_radius

        this.lon_load_transfer = lon_load_transfer
        this.lat_load_transfer = lat_load_transfer

        this.final_drive_ratio = final_drive_ratio
        this.motor_efficiency = motor_efficiency
        this.drive_efficiency = drive_efficiency

        this.drag_area = drag_area
        this.downforce_area = downforce_area

        this.max_torque = max_torque
        this.max_power_per_soc = max_power_per_soc
        this.capacity = capacity

        this.tire_circumference = wheel_radius * np.radians(360)

        this.AWD = AWD

        # END

    

    def max_accel_g(this, velocity : float, soc : float):
        motor_rps = (velocity / this.tire_circumference) * this.final_drive_ratio
        if motor_rps * 60 > 6500: return 0
        
        d_force = this.get_downforce(velocity) + (this.mass * 9.81)

        max_motor_torque = min(this.max_torque, this.max_power_per_soc.f(soc) * 9.54929677 / (motor_rps * 60))
        max_motor_force = max_motor_torque * this.final_drive_ratio / this.wheel_radius
        max_motor_force *= this.drive_efficiency
        max_motor_force -= this.get_drag(velocity)
        max_motor_g = max_motor_force / (this.mass * 9.81)
       
        max_tire_g = 0

        i = 0
        while(max_tire_g < max_motor_g and i < 5):
            lon_pos = this.lon_load_transfer.f(max_tire_g)
            lon_neg = 1 - this.lon_load_transfer.f(max_tire_g)
            lat_pos = this.lat_load_transfer.f(0)
            lat_neg = 1 - this.lat_load_transfer.f(0)

            normal_force_rr = lon_neg * lat_pos * d_force
            normal_force_lr = lon_neg * lat_neg * d_force
            normal_force_rf = lon_pos * lat_pos * d_force
            normal_force_lf = lon_pos * lat_neg * d_force

            max_tire_force = this.coeff_fric_lon.f(normal_force_rr) * normal_force_rr
            max_tire_force += this.coeff_fric_lon.f(normal_force_lr) * normal_force_lr
            if this.AWD:
                max_tire_force += this.coeff_fric_lon.f(normal_force_rf) * normal_force_rf
                max_tire_force += this.coeff_fric_lon.f(normal_force_lf) * normal_force_lf
            max_tire_g = max_tire_force / (this.mass * 9.81)
            
            i += 1

        return min(max_motor_g, max_tire_g)
         
    def get_drag(this, velocity):
        return (this.downforce_area * (velocity**2))

    def get_tire_f(this, normal_force):
        return this.coeff_fric_lon  

    def get_downforce(this, velocity):
        return (this.downforce_area * (velocity**2))

    def plot_elipse(this, velocity : float):
        t = np.linspace(0, 360, 360)
        
        df_mult = ((this.mass * 9.81) + this.get_downforce(velocity)) / (this.mass * 9.81)

        x = this.coeff_fric_lat.f(df_mult / 2) * df_mult * np.cos(np.radians(t))
        y = this.coeff_fric_lon.f(df_mult / 2) * df_mult * np.sin(np.radians(t))

        mv = this.max_accel_g(velocity, 100)

        y = np.array(list(map((lambda n : min(n, mv)), y)))

        plt.plot(x,y, color=matplotlib.colors.hsv_to_rgb(((velocity / 60.0), 1, 1)))

if __name__ == '__main__':
    v = vehicle(280, 2, 1.5, polynomial([-0.0001, 1.7]), polynomial([1.25]), 0.22,
                polynomial([0.5]), polynomial([0.5]), 
                4.0, polynomial([0.94]), 0.98, 
                0.5, 0.5, 
                300, polynomial([80000]), 6.2, #[7.01587e-7, 0, 0.0793916, -19.2291, 1525.91552, 0]), 6.2
                )
    
    plt.gca().set_aspect('equal')
    for i in range(0, 9):
        v.plot_elipse(float(i * 5) + 2.43) #7312455278365)

    plt.title("Max accleration at speed")
    plt.show()
