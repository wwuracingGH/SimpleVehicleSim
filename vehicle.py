# simple dynamic model of a vehicle

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sc

import math, matplotlib
AIRDENSITY = 1.204
GRAV       = 9.806


class polynomial:
    def __init__(this, values : list[float]):
        this.values = values
        this.max_power = len(values)
    
    def f(this, x):
        n = this.values[-1]
        for i in range(1, this.max_power):
            n += this.values[- (i + 1)] * x ** i
        return n

v67_battery_r = polynomial([2.2388096172e-8,-0.00000491854642229,0.000367215355266,-0.0112200142003,0.340835479399])

def get_battery_r(soc_percent, temp_c):
    return v67_battery_r.f(soc_percent) * ((73.52577 / temp_c) - 0.927307)

class lookuptable_2D:
    def __init__(this, values : list[list[float]], xmin=0, xmax=1, ymin=0, ymax=1):
        this.spline = sc.RegularGridInterpolator((np.linspace(1,0,len(values[0])), np.linspace(1,0,len(values))), np.transpose(values), method='linear') 
        this.xmin, this.xmax, this.ymin, this.ymax = xmin, xmax, ymin, ymax

    # basic bilinear interpolation
    def f(this, x, y):
        xv = (x - this.xmin) / (this.xmax - this.xmin)
        yv = (y - this.ymin) / (this.ymax - this.ymin)
        return this.spline((np.minimum(1+yv*0, np.maximum(yv, yv*0)), np.minimum(1+xv*0, np.maximum(xv, xv*0))))

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
            pack_voltage_curve  : polynomial,
            max_regen_watts     : polynomial,     
            regen_mix           : float,          # the percentage of braking torque that goes into regen
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
        this.max_regen_power = max_regen_watts
        this.regen_mix = regen_mix
        this.pack_voltage = pack_voltage_curve

        this.drag_area = drag_area
        this.downforce_area = downforce_area

        this.max_torque = max_torque
        this.max_power_per_soc = max_power_per_soc
        this.capacity = capacity

        this.tire_circumference = wheel_radius * np.radians(360)
        this.max_speed = this.motor_efficiency.xmax / (60 * final_drive_ratio) * this.tire_circumference

        this.AWD = AWD

        # END

    def max_accel_g(this, velocity : float, soc : float, power_curve:polynomial=None):
        if velocity > this.max_speed: 
            return 0, 0
        motor_rpm = (velocity / this.tire_circumference) * this.final_drive_ratio * 60

        if power_curve is None:
            power_curve = this.max_power_per_soc

        max_tire_g = this.max_tire_g(velocity, False)
        max_tire_force = max_tire_g * (this.mass * GRAV)
        max_tire_torque = max_tire_force * this.wheel_radius / this.final_drive_ratio
        max_battery_torque = (power_curve.f(soc) * 9.54929677 / motor_rpm) * this.drive_efficiency
        if not this.AWD:
            max_battery_torque = (power_curve.f(soc) * 9.54929677 / motor_rpm) * this.drive_efficiency * this.motor_efficiency.f(motor_rpm, min(max_battery_torque, max_tire_torque, this.max_torque))
            max_motor_torque = min(this.max_torque, max_battery_torque)
        else:
            max_battery_g = (power_curve.f(soc) / velocity) / (this.mass * GRAV) * this.drive_efficiency
            max_g = min(max_battery_g, max_tire_g)

            rear_balance = this.cg_bal * (this.cg_height / this.wheelbase) * max_g
            total_torque = ((max_g * GRAV) * this.mass * velocity) * 9.54929677 / motor_rpm
            rear_motor_torque = min(total_torque * rear_balance * 0.5, this.max_torque)
            front_motor_torque = min(total_torque * (1-rear_balance) * 0.5, this.max_torque)

            if max_battery_g == max_g:
                rear_motor_torque *= this.motor_efficiency.f(motor_rpm, rear_motor_torque)
                front_motor_torque *= this.motor_efficiency.f(motor_rpm, front_motor_torque)

            max_motor_torque = (rear_motor_torque + front_motor_torque) * 2

        max_motor_force = (max_motor_torque) * this.final_drive_ratio / this.wheel_radius
        max_motor_force -= this.get_drag(velocity)
        max_motor_g = max_motor_force / (this.mass * GRAV)

        return min(max_motor_g, max_tire_g), min(max_motor_torque, max_tire_torque)

    def max_tire_g(this, velocity : float, lat=True, braking=False):
        if lat: 
            mu = this.coeff_fric_lat.values[-1]
            sens = 0
            if len(this.coeff_fric_lat.values) == 2:
                sens = this.coeff_fric_lat.values[0]
        else:
            mu = this.coeff_fric_lon.values[-1]
            sens = 0
            if len(this.coeff_fric_lon.values) == 2:
                sens = this.coeff_fric_lon.values[0]

        full_tires = (lat or this.AWD or braking)

        h = this.cg_height
        if lat:
            b = this.trackwidth
        else:
            b = this.wheelbase
        cv2 = this.get_downforce(velocity)

        if sens == 0 and full_tires:
            return (mu * this.mass * GRAV + 2 * mu * cv2) / (this.mass * GRAV)
        elif sens == 0:
            return mu * b * (this.mass * GRAV + 2 * cv2) / (2 * this.mass * (b - h * mu) * GRAV)

        msg = this.mass * GRAV * sens
        hsq = this.cg_height * this.cg_height

        if full_tires:
            root = (b * b) - (16 * hsq * sens * cv2 * (mu + sens * cv2 + msg)) - (4 * hsq * msg * (msg + (2 * mu)))
            root = (b - math.sqrt(root)) * b
            div = 4 * hsq * msg
        else:
            root = b * (b - 2 * h * (2 * sens * cv2 + msg + mu)) + hsq * mu * mu
            root = b * (b - h * (msg + 2 * sens * cv2 + mu) - math.sqrt(root))
            div = 2 * hsq * msg

        return root / div

    def max_lat_accel_g(this, velocity : float):
        return this.max_tire_g(velocity, True) 

    def max_braking_accel_g(this, velocity):
        return this.max_tire_g(velocity, False, True) 

    def max_velocity_of_c(this, curvature):
        c = max(this.coeff_fric_lat.values[-1] * GRAV/(this.max_speed**2), abs(curvature))

        radius = 1 / c

        velocity = math.sqrt(this.max_lat_accel_g(0) * GRAV * radius)
        for _ in range(0, 10):
            velocity = math.sqrt(this.max_lat_accel_g(velocity) * GRAV * radius)
        return velocity 
 

    def rpm_from_velocity(this, velocity):
        wheel_rps = velocity / (this.wheel_radius * 2 * math.pi)
        return wheel_rps * 60 * this.final_drive_ratio

    def drivetrain_efficiency_at(this, acceleration, velocity):
        rpm = this.rpm_from_velocity(velocity)
        f_a = (acceleration * this.mass) * (1 / this.drive_efficiency) 
        if not this.AWD:
            torque = (f_a / this.final_drive_ratio) * this.wheel_radius
            m_e = this.motor_efficiency.f(rpm, torque)
        else:
            rear_balance = this.cg_bal * (this.cg_height / this.wheelbase) * (acceleration / GRAV)
            torque_r = ((f_a * rear_balance) / this.final_drive_ratio) * this.wheel_radius
            torque_f = (f_a * (1 - rear_balance) / this.final_drive_ratio) * this.wheel_radius
            m_e = this.motor_efficiency.f(rpm, torque_r) * rear_balance + this.motor_efficiency.f(rpm, torque_f) * (1 - rear_balance)

        return m_e * this.drive_efficiency 

    def energy_used_at(this, acceleration, velocity, soc, dt, temp_c=30):
        if acceleration > 0: 
            power = this.mass * 2 * acceleration * (velocity) * (1 / this.drivetrain_efficiency_at(acceleration, velocity))
        else:
            power = this.mass * 2 * abs(acceleration) * (velocity) * (1 / this.drivetrain_efficiency_at(abs(acceleration), velocity)) * this.regen_mix
            power = min(power, this.max_regen_power.f(soc))
            power = math.copysign(power, acceleration)

        open_voltage = this.pack_voltage.f(soc)
        r_pack = get_battery_r(soc, temp_c)
        current = (open_voltage - math.sqrt(open_voltage**2 - 4 * abs(power) * r_pack)) / (2 * r_pack)
        #print(f'res: {r_pack}\ttemp: {temp_c}\tsoc: {soc}')
        excess_power = current**2 * r_pack

        return power * dt, excess_power * dt
                     

    def get_drag(this, velocity):
        return (this.drag_area * (velocity**2) * 1.2)

    def get_tire_f(this, normal_force):
        return this.coeff_fric_lon  

    def get_downforce(this, velocity):
        return (this.downforce_area * (velocity**2) * AIRDENSITY)

    def plot_elipse(this, ax):
        t = np.linspace(0, 360, 60)
        velocity = np.linspace(0.001, 6500 / (60 * this.final_drive_ratio) * this.tire_circumference, 20)

        df_mult = ((this.mass * GRAV) + this.get_downforce(velocity)) / (this.mass * GRAV)
        
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
    v = VEHICLE_V67 

    x = np.linspace(0, v.max_speed, 30)
    y = [v.max_lat_accel_g(xh) for xh in x]
    plt.plot(x, y)
    plt.show()

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
   
    #v.plot_elipse(ax)

    #plt.title("Max accleration at speed and SOC")
    #plt.show()
