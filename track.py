import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

TRACKDENSITY = 4 #how many line segments per meter

BEZIERDENSITY = 200 #how many points are measured on each bezier

def lerp(a, b, t):
    return (1-t) * a + t * b

class segment:
    def __init__(self, point, curvature, normal, dist = 0):
        self.p = point
        self.c = curvature
        self.n = normal
        self.s = dist

class trackDef:
    def __init__(self, segments, length, halfwidth):
        self.segments = segments
        self.length = ((len(segments) - 1) / TRACKDENSITY)
        self.width = halfwidth

    #from 0 to 1
    def getDistance(self, s):
        return s * self.length

    #d is in meters
    def getPoint(self, d):
        i = math.floor(d * TRACKDENSITY)
        t = (d * TRACKDENSITY) - i

        return (lerp(self.segments[i].p[0], self.segments[i+1].p[0], t),
                lerp(self.segments[i].p[1], self.segments[i+1].p[1], t))

    def getCurve(self, d):
        i = math.floor(d * TRACKDENSITY)
        t = (d * TRACKDENSITY) - i
        return lerp(self.segments[i].c, self.segments[i+1].c, t)

    def getNormal(self, d):
        i = math.floor(d * TRACKDENSITY)
        t = (d * TRACKDENSITY) - i
        return normalize(lerp(self.segments[i].n[0], self.segments[i+1].n[0], t),
                lerp(self.segments[i].n[1], self.segments[i+1].n[1], t))

    def getAngle(self, d):
        x, y = getNormal(d)
        return math.atan2(y, x)

    def toCartesian(self, s, n):
        px, py = self.getPoint(s)
        oy, ox = self.getNormal(s)
        return (px - (ox * n), py + (oy * n))

    def getBoundsPyplot(self):
        boundsxi = []
        boundsxo = []
        boundsyi = []
        boundsyo = []

        for seg in self.segments:
            nx, ny = seg.n
            nx *= self.width
            ny *= self.width
            px, py = seg.p
            boundsxo.append((px - ny) - self.segments[0].p[0])
            boundsyo.append((py + nx) - self.segments[0].p[1])
            boundsxi.append((px + ny) - self.segments[0].p[0])
            boundsyi.append((py - nx) - self.segments[0].p[1])
        
        return (boundsxi, boundsyi, boundsxo, boundsyo)

    # returns list of distances
    def getCurvePoints(self):
        is_in_curve = False
        current_distance = 0
        center_distances = []

        curve_start_distance = 0
        av_curvature = 0

        for seg in self.segments:
            if abs(seg.c) < 0.08:
                if is_in_curve:
                    is_in_curve = False
                    delta = current_distance - curve_start_distance
                    center_distances.append((curve_start_distance, curve_start_distance + (delta * 0.5), current_distance, av_curvature))
            else:
                if not is_in_curve:
                    is_in_curve = True
                    av_curvature = seg.c
                    curve_start_distance = current_distance

            current_distance += 1 / TRACKDENSITY

        return center_distances
    
    def get_curvetrack(self):
        segs = []
        for l,r in self.defs:
            sw = l/r if r != 0 else l
            turn = 1 if r > 0 else 0 if r < 0 else -1

            segs.append([1 if turn != -1 else 0,abs(sw),abs(r),turn])
        return segs

def normalize(x,y):
    return (x / (x*x + y*y) ** 0.5, y / (x*x + y*y) ** 0.5)

def distance2D(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx*dx + dy*dy)**0.5

def cBezierPoint(ps, t):
    u = (1-t)
    x = u*u*u*ps[0][0] + 3*u*u*t*ps[1][0] + 3*u*t*t*ps[2][0] + t*t*t*ps[3][0] 
    y = u*u*u*ps[0][1] + 3*u*u*t*ps[1][1] + 3*u*t*t*ps[2][1] + t*t*t*ps[3][1]
    return (x, y)

def cBezierNormal(ps, t):
    u = (1-t)
    x = 3*u*u*(ps[1][0] - ps[0][0]) + 6*u*t*(ps[2][0] - ps[1][0]) + 3*t*t*(ps[3][0] - ps[2][0])
    y = 3*u*u*(ps[1][1] - ps[0][1]) + 6*u*t*(ps[2][1] - ps[1][1]) + 3*t*t*(ps[3][1] - ps[2][1])

    return normalize(x, y)

def cBezierCurvature(ps, t):
    u = 1 - t
    xp = 3*u*u*(ps[1][0] - ps[0][0]) + 6*u*t*(ps[2][0] - ps[1][0]) + 3*t*t*(ps[3][0] - ps[2][0])
    yp = 3*u*u*(ps[1][1] - ps[0][1]) + 6*u*t*(ps[2][1] - ps[1][1]) + 3*t*t*(ps[3][1] - ps[2][1])
    xpp = 6*u*(ps[2][0] - 2*ps[1][0] + ps[0][0]) + 6*t*(ps[3][0] - ps[2][0] + ps[1][0]) 
    ypp = 6*u*(ps[2][1] - 2*ps[1][1] + ps[0][1]) + 6*t*(ps[3][1] - ps[2][1] + ps[1][1])

    return ((xp * ypp) - (yp * xpp)) / ((xp * xp + yp * yp) ** (3/2))

# for p in bpy.data.objects[0].data.splines[0].bezier_points:
#     print(str(p.co.x) + "," + str(p.co.y) + "," + str(p.handle_right.x) + "," + str(p.handle_right.y))
#     print(str(p.co.x) + "," + str(p.co.y) + "," + str(p.handle_left.x) + "," + str(p.handle_left.y), end=",")

def trackFromBezierCSV(fp, width, scale_factor=1.0):
    segments = []
    lastdist = 0
    lastPushedDist = 0
    lastnorm = ()

    with open(fp) as f:
        jjj = -1
        while line := f.readline(): 
            jjj += 1
            pv = line.replace('\n', '').split(',')
            bp = []
            for i in range(0, len(pv), 2):
                bp.append((float(pv[i]) * scale_factor, float(pv[i+1]) * scale_factor))

            if(len(segments) == 0):
                segments.append(segment(cBezierPoint(bp, 0), cBezierCurvature(bp, 0), cBezierNormal(bp, 0)))
                lastPushedDist = 0
                lastPushedPoint = cBezierPoint(bp, 0)
                lastnorm = cBezierNormal(bp, 0)

            lastpoint = bp[0]
            for i in range(1, BEZIERDENSITY + 1, 1):
                t = float(i)/BEZIERDENSITY
                np = cBezierPoint(bp, t)
                nd = lastdist + distance2D(np, lastpoint) 
                nn = cBezierNormal(bp, t)

                if((distdelta1 := nd - lastPushedDist - (1/TRACKDENSITY)) >= 0.0):
                    nt = (i - (distdelta1 / (nd - lastdist))) / BEZIERDENSITY

                    curve = math.asin(lastnorm[1] * nn[0] - lastnorm[0] * nn[1]) / (nd - lastdist)
                    point = cBezierPoint(bp, nt)
                    norm  = cBezierNormal(bp, nt)

                    segments.append(segment(point, curve, norm, nd))

                    lastPushedDist = float(len(segments) - 1) / TRACKDENSITY
                    
                lastnorm = nn
                lastpoint = np
                lastdist = nd
    
    return trackDef(segments, lastdist, width)

def trackToMatlab(track, fp):
    with open(fp, 'w') as fp:
        fp.write("x,y,k,s,dk\n")
        
        last_s = 0
        for s in track.segments:
            fp.write(str(s.p[0]) + ',' + str(s.p[1]) + ',' + str(s.c) + ',' + str(s.s) + ',' + str((s.s - last_s) * TRACKDENSITY) + '\n')
            last_s = s.s
            

def toTangentCurve(track, one_way = False):
    lastnorm = track.segments[0].n
    lastpos = track.segments[0].p
    lastcurve = track.segments[0].c

    lastheading = math.acos(lastnorm[0])

    # stored as l, r
    defs = []

    startpos = track.segments[0].p
    startnorm = (0,0)

    straights = []
    isinstraight = False
    straightStart = (0,0)

    for i,s in enumerate(track.segments):
        pos = s.p
        norm = s.n
        curve = s.c 

        heading = math.acos(norm[0])

        if abs(curve) < 0.022 or (curve * lastcurve) <= 0:
            if isinstraight:
                pass  
            else:
                isinstraight = True
                straightStart = pos

            lastheading = heading
            
            lastnorm = norm
            lastpos = pos
            lastcurve = curve
        else:
            if isinstraight:
                isinstraight = False
                if len(straights) == 0:
                    lengthofnorm = ((pos[0] - straightStart[0]) ** 2 + (pos[1] - straightStart[1]) ** 2) ** 0.5
                    startnorm = ((pos[0] - straightStart[0]) / lengthofnorm, (pos[1] - straightStart[1]) / lengthofnorm) 
                straights.append([straightStart, pos])

    for i in range(len(straights) - (1 if one_way else 0)):
        p0, p1 = straights[i][0], straights[i][1]
        p2, p3 = straights[i - len(straights) + 1][0], straights[i - len(straights) + 1][1]
        norm1 = (p1[0] - p0[0], p1[1] - p0[1])
        norm2 = (p3[0] - p2[0], p3[1] - p2[1])

        d_seg1 = (norm1[0]**2 + norm1[1]**2) ** 0.5
        d_seg2 = (norm2[0]**2 + norm2[1]**2) ** 0.5

        norm1 = (norm1[0]/d_seg1, norm1[1]/d_seg1)
        norm2 = (norm2[0]/d_seg2, norm2[1]/d_seg2)

        angle = math.atan2(norm1[1],norm1[0]) - math.atan2(norm2[1],norm2[0])
        if abs(angle) > math.pi:
            angle = (2 * math.pi - abs(angle)) * angle/abs(angle)
        
        slope1 = norm1[1]/norm1[0]
        slope2 = norm2[1]/norm2[0]
        slope3 = - 1.0 / slope1 
        slope4 = - 1.0 / slope2

        x_intersection = ((p1[1] - p2[1]) + (slope2 * p2[0] - slope1 * p1[0])) / (slope2 - slope1)
        distA = (x_intersection - p1[0]) / norm1[0]
        distB = (x_intersection - p2[0]) / norm2[0]

        mindist = min(abs(distA), abs(distB))
        p4 = ((p1[0] + (distA-mindist) * norm1[0]), (p1[1] + (distA-mindist) * norm1[1]))
        p5 = ((p2[0] + (distB+mindist) * norm2[0]), (p2[1] + (distB+mindist) * norm2[1]))

        x_center = ((p4[1] - p5[1]) + (slope4 * p5[0] - slope3 * p4[0])) / (slope4 - slope3)
        r = (x_center - p4[0]) / norm1[1] 
        l = abs(angle * r)

        straights[i - len(straights) + 1][0] = p5

        distline = ((p4[0] - p0[0])**2 + (p4[1] - p0[1])**2) ** 0.5

        defs.extend([(distline, 0), (l, r)])

    segments, total_dist = segs_from_defs(defs, startpos, startnorm)
    track2 = trackDef(segments, 0, track.width)

    print("distance traveled in 1 lap: " + str(total_dist))
    track2.defs = defs
    
    return track2 

def segs_from_defs(defs, startpos, startnorm):  
    segments = []
    lastpos = startpos
    lastnorm = startnorm

    total_dist = 0

    segments.append(segment(startpos, 0, startnorm, 0))

    for _i,(l,r) in enumerate(defs):
        print('length: ' + str(round(l * 1000) / 1000) + '\tradius: ' + str(round(r * 1000) / 1000))
        lp = max(math.ceil(l * 4), 8)
        dl = l / lp

        typ = 0 if r < 0 else 1 if r > 0 else -1
        sw = l if r == 0 else (l/r)

        if r == 0:
            firstpos = lastpos
            for i in range(1, lp + 1):
                lastpos = ((firstpos[0] + (lastnorm[0] * i * dl)), (firstpos[1] + (lastnorm[1] * i * dl)))
                segments.append(segment(lastpos, 0, lastnorm, total_dist + dl * i))
                #segments[-1].c = (0 + (1.0/defs[_i - len(defs) + 1][1])) / 2.0
        else:
            center = ((lastpos[0] + (r * lastnorm[1])), (lastpos[1] - (r * lastnorm[0])))
            sangle = math.atan2(lastnorm[0], -lastnorm[1])

            for i in range(1, lp + 1):
                angle = sangle - (i * dl / r)
                lastpos = (center[0] + (math.cos(angle) * r), center[1] + (math.sin(angle) * r))
                lastnorm = (math.sin(angle), -math.cos(angle))
                segments.append(segment(lastpos, -1.0/r, lastnorm, total_dist + dl * i))
                next_r = defs[_i - len(defs) + 1][1]
                #segments[-1].c = ((1.0/r) + (1.0/next_r if next_r != 0 else 0)) / 2.0
        total_dist += l
        
    return segments, total_dist

def objective_func(track):
    x_point = track.segments[0].p[0]
    y_point = track.segments[0].p[1]
    
    print(x_point, y_point)
    xp, yp = [x_point], [y_point] 
    
    theta = math.atan2(track.segments[0].n[1], track.segments[0].n[0])
    
    for i in range(1, len(track.segments)):  
        ds = track.segments[i].s - track.segments[i - 1].s
        
        av_c = track.segments[i].c # (track.segments[i].c + track.segments[i - 1].c) * 0.5
        theta += av_c * ds
        
        dx = math.cos(theta) * ds
        dy = math.sin(theta) * ds
        
        x_point += dx
        y_point += dy
        
        xp.append(x_point)
        yp.append(y_point)
        
        track.segments[i].p = (x_point, y_point)
    
    return (xp, yp) 

def parse_csv(fp):
    cols = {}
    with open(fp) as f:
        colnames = f.readline().replace('\n', '').split(',')
        for c in colnames:
            print(c)
            cols[c] = []
        
        line = f.readline()
        
        while len(line) > 2:
            dat = line.replace('\n', '').split(',')
            for i,c in enumerate(dat):
                cols[colnames[i]].append(float(c))
            line = f.readline()
        
    return cols 

def TrackFromLR(fp, width):
    d = parse_csv(fp)
    print(d.keys())
    defs = list(zip(d['l'], d['r']))
    segments, dist = segs_from_defs(defs, (0,0), (1, 0))
    return trackDef(segments, dist, width)
 
if __name__ == '__main__':
    track = trackFromBezierCSV("defaulttrack.csv", 1.5, 1.511)
    print(track.length)
    
    plt.gca().set_aspect('equal')
    trackToMatlab(track, "testout.csv")
    
    xp, yp = objective_func(track)
    plt.plot(xp, yp)
    #track = TrackFromLR('autoXlr.csv', 1.5)
#    bxi, byi, bxo, byo = track.getBoundsPyplot()



#    plt.plot(bxi, byi, c='black')
#    plt.plot(bxo, byo, c='black')

#    track2 = toTangentCurve(track, False)
#    axi, ayi, axo, ayo = track2.getBoundsPyplot()

    #rcx, rcy, cs = track2.getRacingLine(1000)
    #plt.scatter(rcx, rcy, c=cs)
    
#    plt.plot(axi, ayi, c='blue')
#    plt.plot(axo, ayo, c='blue')

#    print()
    
#    xp, yp = objective_func(track2)
#    plt.plot(xp, yp)
    
#    trackToMatlab(track2, 'testout.csv')

    

    #plt.scatter(px, py, c=col)
    #for i in range(len(col)):
    #    plt.annotate(str(float(int(10/col[i]))/10), (px[i], py[i]))

    plt.show()
