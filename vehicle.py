# simple dynamic model of a vehicle

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sc

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

class lookuptable_2D:
    def __init__(this, values : list[list[float]], xmin=0, xmax=1, ymin=0, ymax=1):
        this.spline = sc.RegularGridInterpolator((np.linspace(1,0,len(values[0])), np.linspace(1,0,len(values))), np.transpose(values), method='linear') 
        this.xmin, this.xmax, this.ymin, this.ymax = xmin, xmax, ymin, ymax

    # basic bilinear interpolation
    def f(this, x, y):
        xv = (x - this.xmin) / (this.xmax - this.xmin)
        yv = (y - this.ymin) / (this.ymax - this.ymin)
        return this.spline((yv,xv))

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
            cg_height           : float,
            cg_bal              : float,      # often called 'a'
            # Drivetrain
            final_drive_ratio   : float,
            pack_efficiency     : lookuptable_2D,
            motor_efficiency    : lookuptable_2D, # part of motor efficiency per torque
            drive_efficiency    : float,
            max_regen_watts     : polynomial,     # state of charge lookup table basically
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

        this.cg_height = cg_height
        this.cg_bal = cg_bal

        this.final_drive_ratio = final_drive_ratio
        this.pack_efficiency  = pack_efficiency
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
        
        d_force = this.get_downforce(velocity) + (this.mass * 9.806)

        i = 0
        g_t = 1.4
        max_tire_g = (g_t * this.cg_bal) / (1 - (g_t * this.cg_height / this.wheelbase))
        while(i < 10):
            lon_pos = (max_tire_g * this.cg_height / this.wheelbase) + this.cg_bal
            lon_neg = 1 - lon_pos

            normal_force_rr = lon_neg * d_force

            g_t = this.coeff_fric_lon.f(normal_force_rr / 2)
            g_t2 = this.coeff_fric_lon.f(normal_force_rr / 2)
            max_tire_g = (g_t * this.cg_bal) / (1 - (g_t * this.cg_height / this.wheelbase))
            if (this.AWD): 
                max_tire_g += (g_t2 * (1 - this.cg_bal)) / (1 + (g_t2 * this.cg_height / this.wheelbase))
            max_tire_force = max_tire_g * (this.mass * 9.806)
            i += 1

        max_tire_g = max_tire_force / (this.mass * 9.806)
        max_tire_torque = max_tire_force * this.wheel_radius / this.final_drive_ratio
        max_battery_torque = (this.max_power_per_soc.f(soc) * 9.54929677 / (motor_rps * 60))
        max_motor_torque = min(this.max_torque, max_battery_torque) * this.drive_efficiency * this.motor_efficiency.f(motor_rps * 60, min(min(max_battery_torque, max_tire_torque), this.max_torque))

        max_motor_force = max_motor_torque * this.final_drive_ratio / this.wheel_radius
        max_motor_force -= this.get_drag(velocity)
        max_motor_g = max_motor_force / (this.mass * 9.806)

        return min(max_motor_g, max_tire_g), min(max_motor_torque, max_tire_torque)


    def max_lat_accel_g(this, velocity : float):
        d_force = this.get_downforce(velocity) + (this.mass * 9.806)

        g_t = this.coeff_fric_lat.f(d_force / 4)
        max_tire_g = (g_t) / (1 - (g_t * this.cg_height / this.wheelbase))
       
        for _ in range(0, 10):
            lat_pos = (max_tire_g * this.cg_height / this.trackwidth)

            normal_force_l = lat_pos * this.mass * 9.806 + (this.get_downforce(velocity) / 2)
            normal_force_r = lat_pos * this.mass * 9.806 + (this.get_downforce(velocity) / 2) 

            g_t = this.coeff_fric_lat.f(normal_force_l / 2)
            g_t2 = this.coeff_fric_lat.f(normal_force_r / 2)
            max_tire_g = (g_t * 0.5) / (1 - (g_t * this.cg_height / this.wheelbase))
            max_tire_g += (g_t2 * 0.5) / (1 + (g_t2 * this.cg_height / this.wheelbase))

        return max_tire_g

    def max_braking_accel_g(this, velocity):
        max_tire_g = 0

        d_force = this.get_downforce(velocity) + (this.mass * 9.81)
        
        i = 0
        while(i < 5):
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
            max_tire_force += this.coeff_fric_lon.f(normal_force_rf) * normal_force_rf
            max_tire_force += this.coeff_fric_lon.f(normal_force_lf) * normal_force_lf
            max_tire_g = max_tire_force / (this.mass * 9.81)
            
            i += 1

        return max_tire_g

    def rpm_from_velocity(this, velocity):
        wheel_rps = velocity / (this.wheel_radius * 2 * math.pi)
        return wheel_rps * 60 * this.final_drive_ratio

    def drivetrain_efficiency_at(this, acceleration, velocity):
        rpm = this.rpm_from_velocity(velocity)
        f_a = (acceleration * this.mass) * (1 / this.drive_efficiency) 
        torque = (f_a / this.final_drive_ratio) * this.wheel_radius
        m_e = this.motor_efficiency.f(rpm, torque)
        return m_e * this.drive_efficiency 

    def get_drag(this, velocity):
        return (this.drag_area * (velocity**2) * 1.2)

    def get_tire_f(this, normal_force):
        return this.coeff_fric_lon  

    def get_downforce(this, velocity):
        return (this.downforce_area * (velocity**2))

    def plot_elipse(this, ax):
        t = np.linspace(0, 360, 60)
        velocity = np.linspace(0.001, 6500 / (60 * this.final_drive_ratio) * this.tire_circumference, 20)

        df_mult = ((this.mass * 9.81) + this.get_downforce(velocity)) / (this.mass * 9.81)
        
        xvals = []
        yvals = []
        zvals = []

        for v in velocity:
            x = this.max_lat_accel_g(v) * np.cos(np.radians(t))
            y = this.max_braking_accel_g(v) * np.sin(np.radians(t))
            mv = this.max_accel_g(v, 100)
            y = np.array(list(map((lambda n : min(n, mv)), y)))
            xvals.append(x)
            yvals.append(y)
            zvals.append([v] * len(x))

        ax.scatter(xvals,yvals,zvals)

    def plot_powersurface(this, ax):
        soc = np.linspace(5, 100, 20)
        velocity = np.linspace(0.001, 6500 / (60 * this.final_drive_ratio) * this.tire_circumference, 250)
        
        x, y = np.meshgrid(velocity, soc)
        z = np.array([[this.max_accel_g(v,c) for v in velocity] for c in soc])

        ax.set_xlabel("speed (m/s)")
        ax.set_ylabel("state of charge (%)")
        ax.set_zlabel("accel (G)")
        ax.plot_surface(x, y, z, rstride=20, cstride=10, cmap=matplotlib.cm.coolwarm)

if __name__ == '__main__':
    v = vehicle(280, 2, 1.5, polynomial([-0.0001, 1.5]), polynomial([1.25]), 0.23,
                polynomial([0.5]), polynomial([0.5]), 
                3.3, polynomial([0.94]), 0.98, 
                0.5, 0.5, 
                300, polynomial([7.01587e-7, 0, 0.0793916, -19.2291, 1525.91552, 0]), 6.2, True
                )
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
   
    max_speed = 6500 / (60 * v.final_drive_ratio) * v.tire_circumference

    v.plot_elipse(ax)

    plt.title("Max accleration at speed and SOC")
    plt.show()
