from math import sqrt, log, factorial, exp
from numpy import arange, array
#%matplotlib qt
from matplotlib.pyplot import subplots, subplots_adjust, subplot, figure, \
                              axes, text, bar, plot, legend, grid, title, \
                              xlabel, ylabel, show, ylim
from matplotlib.widgets import Slider, Button, RadioButtons
from datetime import datetime, timedelta 

COUNTY_DATA = {datetime(2020,3,6):[1,0],
               datetime(2020,3,7):[1,0],
               datetime(2020,3,8):[1,0],
               datetime(2020,3,9):[1,0],
               datetime(2020,3,10):[2,0],
               datetime(2020,3,11):[1,0],
               datetime(2020,3,12):[1,0],               
               datetime(2020,3,13):['u',1],
               datetime(2020,3,14):['u',1],
               datetime(2020,3,15):['u',1],
               datetime(2020,3,16):['u',1],
               datetime(2020,3,17):['u',1],
               datetime(2020,3,18):['u',1],
               datetime(2020,3,19):['u',2],
               datetime(2020,3,20):['u',2],
               datetime(2020,3,21):[37,3],
               datetime(2020,3,22):[51,3],
               datetime(2020,3,23):['u',3],
               datetime(2020,3,24):[106,3],
               datetime(2020,3,25):[122,5],
               datetime(2020,3,26):[137,7],
               datetime(2020,3,27):[160,7],
               datetime(2020,3,28):['u',10],
               datetime(2020,3,29):[245,10],
               datetime(2020,3,30):[286,11],
               datetime(2020,3,31):[314,13],
               datetime(2020,4,1):['u',14],
               datetime(2020,4,2):[340,16],
               datetime(2020,4,3):[374,18],
               datetime(2020,4,4):[406,22],
               datetime(2020,4,5):[435,25],
               datetime(2020,4,6):[441,28],
               datetime(2020,4,7):[457,28],
               datetime(2020,4,8):[472,30],
               datetime(2020,4,9):[534,32],
               datetime(2020,4,10):[550,33],
               datetime(2020,4,11):['u',35],
               datetime(2020,4,12):[613,37],
               datetime(2020,4,13):[641,39],
               datetime(2020,4,14):[652,41],
               datetime(2020,4,15):[668,43],
               datetime(2020,4,16):[689,48],
               datetime(2020,4,17):[708,49],
               datetime(2020,4,17):[721,49],
               datetime(2020,4,19):[731,49],
               datetime(2020,4,20):[734,50]
               }

def dataHandler(start_date=datetime(2020,3,1), conf_case_delay=7):
    # Handles the El Paso County data from Case_Data.py
    
    death_times=[]; case_times=[]; deaths = []; cases = []
    for date in COUNTY_DATA.keys():
        if not COUNTY_DATA[date][0] == 'u':
            cases.append(COUNTY_DATA[date][0])
        if not COUNTY_DATA[date][1] == 'u':
            deaths.append(COUNTY_DATA[date][1])
    for date in COUNTY_DATA.keys():
        time = (date-start_date).days + (date-start_date).seconds/86400
        if not COUNTY_DATA[date][0] == 'u':
            if time-conf_case_delay >= 0:
                case_times.append(time-conf_case_delay)
                case_times.sort()
            else:
                cases = cases[1:]
        if not COUNTY_DATA[date][1] == 'u':
            death_times.append(time)
            death_times.sort()
    
    return [death_times, case_times, deaths, cases]

def SIR_nu(start_date=datetime(2020,3,1), P=472688, I=1000, R=300, rho=1, tau=5, 
           nu=1.5,loc='Colorado Springs', MaxDays = 100, suppress_output=0,
           rho_sched = {}, d=0, mort_rate=0.02,symp_2_death=15, t=1/48):

    # Handle changing reproduction number over time
    rho_dates = []
    rho_sched[start_date] = rho
    for rho_date in rho_sched.keys():
        rho_dates.append(rho_date); rho_dates.sort()
        if rho_date > start_date + timedelta(days=MaxDays):
            print("Error: A date given in the R0 schedule (%s) exceeds the value for MaxDays ($g)." %
                                             (str(rho_date.date()), MaxDays))
    rho_Ts = []
    prev_rho_MaxDays=0
    rho_dates = rho_dates[1:]
    for rho_date in rho_dates:
        rho_MaxDays = abs(rho_date - start_date).days + abs(rho_date - start_date).seconds/86400
        rho_Ts.append(arange(prev_rho_MaxDays, rho_MaxDays, t))
        prev_rho_MaxDays = rho_MaxDays
    rho_Ts.append(arange(prev_rho_MaxDays, MaxDays, t))
    
    S = P - (I + R)
    
    if suppress_output ==0:
        print("                               Starting Date/Time: " + str(start_date))
        
        print("                      Population of " + loc + ": " + str(P))
        print("           Assumed initial number of infected people: " + str(I))
        print("          Assumed initial number of recovered people: " + str(R))
        print("        Assumed initial number of susceptible people: " + str(S))

        print("      Average # people spread to by infected (rho): " + str(rho))
        print("  Assumed # days a sick person is infectious (tau): " + str(tau))
        print("             Assumed population mixing factor (nu): " + str(nu))

    # Let's normalize P, S, I, and R for simplicity of the equations.
    p = P/P # I know this is obviously 1, just bear with me
    s = S/P
    i = I/P
    r = R/P

    # Here are the differential equations governing the infection dynamics.
    ds_dt = -(rho/tau)*i*(s ** nu) # The two asterisks next to each other is how you write an exponent in python.
    di_dt = (rho/tau)*i*(s ** nu) - (i/tau)
    dr_dt = (i/tau) # It should be noted that this last equation is redundant

    # Let's create a time axis now. We will make it 100 days long, with an interval of 30 minutes.
    T = arange(0,MaxDays,t)

    # We could try to find the integral of s, i, and r analytically, but why would we do that when we have a computer?
    # Let's use a numerical integration technique, such as Euler's method.
    # x_next = x_current + t_interval * dx_dt_current, approximately.

    # First, we need to realize that what we have defined currently for s, i, & p (and their derivatives) 
    # are the starting values, so let's turn them into lists where these values are the starting values.
    p = [p]
    s = [s]
    i = [i]
    r = [r]
    d = [d]
    ds_dt = [ds_dt]
    di_dt = [di_dt]
    dr_dt = [dr_dt]

    # Now we will step through the whole 100 days, appending the new values for p, s, i & r to their respective lists.
    for T in rho_Ts:
        rho = rho_sched[start_date + timedelta(T[0])]
        T = list(T)
        if 0 in T:
            T.remove(0)
        for time in T:  # What is T[1:] ? This means every value in T except for the first one, T[0]. We already have those.
            i.append(i[-1] + t*di_dt[-1])   # What is i[-1]? This means the last value in the list, the 'current' value for this step.
            s.append(s[-1] + t*ds_dt[-1])   # .append() just adds a new value to the end of that list.
            r.append(r[-1] + t*dr_dt[-1])
            if time > (symp_2_death-tau):
                # Deaths are a fraction of the recovered population in the past.
                d.append(r[-int((symp_2_death-tau)/t)]*mort_rate)
            else: # If too early, nobody has died yet.
                d.append(0)
            di_dt.append((rho/tau)*i[-1]*(s[-1] ** nu) - (i[-1]/tau))
            ds_dt.append(-(rho/tau)*i[-1]*(s[-1] ** nu))
            dr_dt.append((i[-1]/tau))
            p.append(i[-1] + s[-1] + r[-1])

    T = [] # Stitch together the timeframes of different rho values
    for rho_T in rho_Ts:
        T = T + list(rho_T)
        
    if suppress_output==0:
        # Now plot all of our data, which now spans 100 days from March 24th.    
        figure()
        plot(T,i, label="Infected")
        plot(T,s, label="Susceptible")
        plot(T,r, label="Recovered")
        legend()
        grid()
        xlabel('Days from ' + str(start_date.date()))
        ylabel('Proportion of Population')

        # So, when is the peak infection date?
        peak_inf_index = i.index(max(i))
        peak_inf_days_from_now = peak_inf_index*t
        peak_date = start_date + timedelta(days=peak_inf_days_from_now)
        thresh_index = i.index(list(filter(lambda k: k > 0.001, i))[0])
        thresh_days_from_now = thresh_index*t
        thresh_date = start_date + timedelta(days=thresh_days_from_now)
        print("                               Peak Infection Date: " + str(peak_date.date()))
        print("                      Peak Infected Simultaneously: " + str(int(max(i)*P)))
        print("                  Proportion Who Will Get Infected: %.1f%%" % (100*r[-1]))
        print("       Date when location will reach 0.1% infected: " + str(thresh_date.date()))
        
    s = array(s)*P  
    i = array(i)*P
    r = array(r)*P
    d = array(d)*P
    return [T,s,i,r,d]
    
def SEIR_nu(start_date=datetime(2020,3,1), P=472688, E=300, I=700, R=300, rho=2.8, 
            tau=5, nu=1.5, mu=3, loc='Colorado Springs', MaxDays = 100, 
            suppress_output=0, rho_sched={}, d=0, mort_rate=0.02,
            symp_2_death=15,t=1/48):

    # Handle changing reproduction number over time
    rho_dates = []
    rho_sched[start_date] = rho
    for rho_date in rho_sched.keys():
        rho_dates.append(rho_date); rho_dates.sort()
        if rho_date > start_date + timedelta(days=MaxDays):
            print("Error: A date given in the R0 schedule (%s) exceeds the value for MaxDays ($g)." %
                                             (str(rho_date.date()), MaxDays))
    rho_Ts = []
    prev_rho_MaxDays=0
    rho_dates = rho_dates[1:]
    for rho_date in rho_dates:
        rho_MaxDays = abs(rho_date - start_date).days + abs(rho_date - start_date).seconds/86400
        rho_Ts.append(arange(prev_rho_MaxDays, rho_MaxDays, t))
        prev_rho_MaxDays = rho_MaxDays
    rho_Ts.append(arange(prev_rho_MaxDays, MaxDays, t))
    
    S = P - (E + I + R)
    
    if suppress_output==0:
        print("                                Starting Date/Time: " + str(start_date))

        print("                      Population of " + loc + ": " + str(P))
        print("           Assumed number of infected people today: " + str(I))
        print("         Assumed number of incubating people today: " + str(E))
        print("          Assumed number of recovered people today: " + str(R))
        print("        Assumed number of susceptible people today: " + str(S))
        
        print("      Average # people spread to by infected (rho): " + str(rho))
        print("                    Assumed incubation period (mu): " + str(mu))
        print("  Assumed # days a sick person is infectious (tau): " + str(tau))
        print("             Assumed population mixing factor (nu): " + str(nu))

    # Let's normalize P, S, I, and R for simplicity of the equations.
    e = [E/P]
    s = [S/P]
    i = [I/P]
    r = [R/P]
    d = [d]

    # Here are the differential equations governing the infection dynamics.
    ds_dt = [-(rho/tau)*i[0]*(s[0] ** nu)] 
    de_dt = [(rho/tau)*i[0]*(s[0] ** nu) - (e[0]/mu)]
    di_dt = [e[0]/mu - (i[0]/tau)]
    dr_dt = [(i[0]/tau)] 

    T = arange(0,MaxDays,t)

    for T in rho_Ts:
        rho = rho_sched[start_date + timedelta(T[0])]
        T = list(T)
        if 0 in T:
            T.remove(0)
        for time in T:  
            s.append(s[-1] + t*ds_dt[-1])
            e.append(e[-1] + t*de_dt[-1])
            i.append(i[-1] + t*di_dt[-1])                               
            r.append(r[-1] + t*dr_dt[-1])
            if time > (symp_2_death-tau):
                # Deaths are a fraction of the recovered population in the past.
                d.append(r[-int((symp_2_death-tau)/t)]*mort_rate)
            else: # If too early, nobody has died yet.
                d.append(0)
            ds_dt.append(-(rho/tau)*i[-1]*(s[-1] ** nu))
            de_dt.append((rho/tau)*i[-1]*(s[-1] ** nu) - e[-1]/mu)
            di_dt.append(e[-1]/mu - (i[-1]/tau))
            dr_dt.append((i[-1]/tau))

    T = [] # Stitch together the timeframes of different rho values
    for rho_T in rho_Ts:
        T = T + list(rho_T)

    if suppress_output==0:
        # Now plot all of our data, which now spans 100 days from March 24th.    
        figure()
        plot(T,i, label="Infected")
        plot(T,s, label="Susceptible")
        plot(T,r, label="Recovered")
        plot(T,e, label="Exposed")
        legend()
        grid()
        xlabel('Days from ' + str(start_date.date()))
        ylabel('Proportion of Population')

        # So, when is the peak infection date?
        peak_inf_index = i.index(max(i))
        peak_inf_days_from_now = peak_inf_index*t
        peak_date = start_date + timedelta(days=peak_inf_days_from_now)
        thresh_index = i.index(list(filter(lambda k: k > 0.001, i))[0])
        thresh_days_from_now = thresh_index*t
        thresh_date = start_date + timedelta(days=thresh_days_from_now)
        print("                               Peak Infection Date: " + str(peak_date.date()))
        print("                      Peak Infected Simultaneously: " + str(int(max(i)*P)))
        print("                  Proportion Who Will Get Infected: %.1f%%" % (100*r[-1]))
        print("        Date when El Paso will reach 0.1% infected: " + str(thresh_date.date()))
    
    s = array(s)*P  
    e = array(e)*P
    i = array(i)*P
    r = array(r)*P
    d = array(d)*P
    return [T,s,e,i,r,d]
    
def NBD_SEIR(start_date=datetime(2020,3,6), p=720403, e=0, i=1, r=0, 
             rho=2.5, tau=5, k=1.5e-1, mu=3, 
             loc='Colorado Springs', MaxDays = 100, 
             suppress_output=0, rho_sched={}, d=0, mort_rate=0.02, 
             symp_2_death=15, t=1/48):
    
    # NBD-SEIR model, taking into account heterogeneous mixing with a 
    # different method than the power law scaling "nu" value proposed
    # by Stroud, et al, instead treating "rho" as a random variable
    # drawn from a Negative Binary Distribution (combination of a 
    # Poisson and a gamma distribution).
    # This model is provided by L. Kong, et al, "Modeling Heterogeneity in 
    # Direct Infectious Disease Transmission in a Compartmental Model"

    # Define:
    # rho = reproductive number. Avg # of secondary infections per capita.
    # tau = average contagious period for an infected person.
    # (gamma = 1/tau; this is the rate of removal of infected)
    # (beta = rho/tau; this is the rate of transmission)
    # k = probability distibution shaping factor for theta.
    # (theta = average number of infected a susceptible with mix with)
    # mu = average incubation period for exposed person.
    # (alpha = 1/mu; this is the rate of exposed becoming infected)
    
    # Number of Susceptible people, as of date
    s = p - (e + i + r)

    # This section handles multiple rho values over time, optionally passed in the rho_sched dictionary
    rho_sched[start_date]=rho
    rho_dates = []
    for rho_date in rho_sched.keys():
        rho_dates.append(rho_date); rho_dates.sort()
        if rho_date > start_date + timedelta(days=MaxDays):
            print("Error: A date given in the R0 schedule (%s) exceeds the value for MaxDays ($g)." %
                                             (str(rho_date.date()), MaxDays))
    rho_Ts = []
    prev_rho_MaxDays = 0
    rho_dates = rho_dates[1:]
    for rho_date in rho_dates:
        rho_MaxDays = abs(rho_date - start_date).days + abs(rho_date - start_date).seconds/86400
        rho_Ts.append(arange(prev_rho_MaxDays, rho_MaxDays, t))
        prev_rho_MaxDays = rho_MaxDays
    rho_Ts.append(arange(prev_rho_MaxDays, MaxDays, t))
    
    
    if suppress_output==0:
        print("                                     Starting Date: " + str(start_date.date()))
        print("                      Population of " + loc + ": " + str(p))
        print("       Assumed initial number of infectious people: " + str(i))
        print("       Assumed initial number of incubating people: " + str(e))
        print("        Assumed initial number of recovered people: " + str(r))
        print("      Assumed initial number of susceptible people: " + str(s))

        # Define the average incubation period. During this time, people are infected, but not infectious.
        #mu = 3

        #print("      Average # people spread to by infected (rho): " + str(rho))
        if len(rho_Ts) > 1:
            for date in rho_sched.keys():
                print("   On %s, the reproduction number (rho) is: %0.1f" % (str(date.date()),rho_sched[date]))
                print("                       transmission rate (beta) is: %0.2f" % (int(rho_sched[date])/tau))
        print("                    Assumed incubation period (mu): " + str(mu))
        print("  Assumed # days a sick person is infectious (tau): " + str(tau))
        print("                     Assumed recovery rate (gamma): " + str(1/tau))
        print("                  Assumed heterogeneity factor (k): " + str(k))
    
    # Calculate other parameters
    beta = rho/tau
    gamma = 1/tau
    
    # Let's normalize P, S, I, and R for simplicity of the equations.
    e = [e]
    s = [s]
    i = [i]
    r = [r]
    d = [d]
    NBD_mean = [beta*i[0]/p]
    m = [k/NBD_mean[-1]]
    NBD_var = [k*(1+m[-1])/(m[-1]**2)]

    # Here are the differential equations governing the infection dynamics.
    ds_dt = [-k*log(1+(rho*i[0])/(tau*k*p))*s[0]] 
    de_dt = [k*log(1+(rho*i[0])/(tau*k*p))*s[0] - (e[0]/mu)]
    di_dt = [e[0]/mu - (i[0]/tau)]
    dr_dt = [(i[0]/tau)] 

    for T in rho_Ts:
        rho = rho_sched[start_date + timedelta(T[0])]
        T = list(T)
        if 0 in T:
            T.remove(0)
        for time in T:  
            #print(time)
            s.append(s[-1] + t*ds_dt[-1])
            e.append(e[-1] + t*de_dt[-1])
            i.append(i[-1] + t*di_dt[-1])                               
            r.append(r[-1] + t*dr_dt[-1])
            if time > (symp_2_death-tau):
                d.append(r[-int((symp_2_death-tau)/t)]*mort_rate)
            else:
                d.append(0)
            ds_dt.append(-k*log(1+(rho*i[-1])/(tau*k*p))*s[-1])
            de_dt.append(k*log(1+(rho*i[-1])/(tau*k*p))*s[-1] - (e[-1]/mu))
            di_dt.append(e[-1]/mu - (i[-1]/tau))
            dr_dt.append((i[-1]/tau))
            NBD_mean.append(beta*i[-1]/p)
            m.append(k/NBD_mean[-1])
            NBD_var.append(k*(1+m[-1])/(m[-1]**2))
        
    T = []    
    for rho_T in rho_Ts:
        T = T + list(rho_T)
    
    peak_date = "Set suppress_output = 0 to see the peak infectious date."
        
    if suppress_output == 0:
        # So, when is the peak infection date?
        peak_inf_index = i.index(max(i))
        peak_inf_days_from_now = peak_inf_index*t
        peak_date = start_date + timedelta(days=peak_inf_days_from_now)
        if max(i)>0.1*p:
            thresh_index = i.index(list(filter(lambda j: j > 0.001*p, i))[0])
            thresh_days_from_now = thresh_index*t
            thresh_date = start_date + timedelta(days=thresh_days_from_now)
            thresh_date = thresh_date.date()
        else:
            thresh_date = "N/A"    
    
        # Now plot all of our data, which now spans 100 days from March 24th.    
        figure()
        plot(T,s, label="Susceptible")
        plot(T,e, label="Exposed")
        plot(T,i, label="Infectious")
        plot(T,r, label="Removed")
        legend()
        grid()
        xlabel('Days from ' + str(start_date.date()))
        ylabel('Population')

        figure()
        plot(T,NBD_mean,label='Mean of NBD')
        plot(T,NBD_var, label='Variance of NBD')
        print("                               Peak Infection Date: " + str(peak_date.date()))
        print("                      Peak Infected Simultaneously: " + str(int(max(i))))
        print("                  Proportion Who Will Get Infected: %.1f%%" % (100*r[-1]/p))
        print("        Date when El Paso will reach 0.1% infected: " + str(thresh_date))

    pdf = [NBD_mean, NBD_var, k, m]
    return [T,s,e,i,r,peak_date,d,pdf]
    
def plot_NBD(NBD_mean=3,k=5,max_value=20,suppress_output=0):

    m = k/NBD_mean
    NBD_var = k*(1+m)/(m**2)
    p = 1/(1+m)
    lamb = k*(p/(1-p))
    def binom(a,b):
        return factorial(a)/(factorial(b)*factorial(a-b))
    X = arange(0,max_value,1)
    P_NBD = []
    P_Pois = []
    for x in X:
        P_NBD.append(binom(x+k-1,x)* (m/(m+1))**k * (1/(m+1))**x)
        P_Pois.append((lamb**x)*(exp(-lamb))/factorial(x))
    p = 1/(1+m)
    lamb = k*(p/(1-p))
    
    if suppress_output==0:
        for i in range(len(X)):
            if P_Pois[i] > P_NBD[i]:
                bar(X[i],P_Pois[i],color='b',alpha=0.4)
                bar(X[i],P_NBD[i],color='r',alpha=0.4)
            else: 
                bar(X[i],P_NBD[i],color='r',alpha=0.4)
                bar(X[i],P_Pois[i],color='b',alpha=0.4)
        bar(0,0,color='b',alpha=0.4,label='Homogeneous Mixing')
        bar(0,0,color='r',alpha=0.4,label='Heterogeneous Mixing')
        legend()
        title("Negative Binomial Distribution; \n \
        Mean = %0.1f / StdDev = %0.1f / k = %0.2f" % (NBD_mean, sqrt(NBD_var), k))
    return [X,P_NBD,P_Pois]
    
def Slider_NBD_SEIR(start_date = datetime(2020,3,1), lock_date = datetime(2020,3,26),
                    post_lock_date = datetime(2020,4,26), init_rho = 4,
                    lock_rho = 1.4, post_lock_rho = 3, init_inf = 40, 
                    conf_case_delay = 7, 
                    mort_rate = 0.013, symp_2_death = 13, MaxDays=350,
                    tau = 8, k = 0.5, nu=1.7, mu = 5.1, model='NBD_SEIR'):
    
    models = {'SIR_nu':0, 'SEIR_nu':1, 'NBD_SEIR':2}
    
    fig, ax = subplots()     
    subplots_adjust(left=0.1, bottom=0.3)
    t = 1/48; today = datetime.now()
    loc = "El Paso County"
    rho_sched = {lock_date:lock_rho, post_lock_date:post_lock_rho}
    lock_time =      (lock_date-start_date).days + \
                     (lock_date-start_date).seconds/86400
    post_lock_time = (post_lock_date-start_date).days + \
                     (post_lock_date-start_date).seconds/86400
    
    if model == 'SIR_nu':
        I_ind = 2; D_ind = 4
        out = SIR_nu(start_date,P=720403,I=init_inf,R=0,
                   rho=init_rho,tau=tau,nu=1.7,loc=loc,MaxDays=MaxDays,suppress_output=1,
                   rho_sched=rho_sched, mort_rate=mort_rate, symp_2_death=symp_2_death,t=t)
    elif model == 'SEIR_nu':
        I_ind = 3; D_ind = 5
        out = SEIR_nu(start_date,P=720403,E=0,I=init_inf,R=0, mu = mu,
                   rho=init_rho,tau=tau,nu=1.7,loc=loc,MaxDays=MaxDays,suppress_output=1,
                   rho_sched=rho_sched, mort_rate=mort_rate, symp_2_death=symp_2_death,t=t)
    elif model == 'NBD_SEIR':
        I_ind = 3; D_ind = 6
        out = NBD_SEIR(start_date,720403,0,init_inf,0,
                   init_rho,tau,k,mu,loc,MaxDays,suppress_output=1,
                   rho_sched=rho_sched, 
                   mort_rate=mort_rate, symp_2_death=symp_2_death)
    
    T = out[0]; I = out[I_ind]; D = out[D_ind]
    
    [death_times, case_times, deaths, cases] = dataHandler(start_date, 
                                                            conf_case_delay)
        
    max_ind = int((today-start_date).days/t)
    # Top-Left Plot
    ax1 = subplot(2,2,1)
    L1, = plot(T[0:max_ind],
               D[0:max_ind],lw=2,label="Predicted Deaths")
    death_scatter1 = plot(death_times,deaths,'ro',label="El Paso Reported Deaths")
    plot([lock_time, lock_time],[0,max(D[0:max_ind])/3],'k')
    text(lock_time-5, max(D[0:max_ind])/3+2, "CO Stay-at-Home order")
    ax1.set_ylim(top=1.1*max(L1.get_ydata()))
    ax1.set_ylim(bottom=0)
    title("Predicted vs. Reported Deaths in El Paso County")
    grid(); legend()
    
    # Top-Right Plot
    ax2 = subplot(2,2,2)
    L2, = plot(T,D,lw=2,label="Predicted Deaths")
    death_scatter2 = plot(death_times,deaths,'ro',label="El Paso Reported Deaths")
    ax2.set_ylim(top=1.1*max(L2.get_ydata()))
    ax2.set_ylim(bottom=0)
    title("Predicted Deaths over the Long Term")
    grid(); legend()
    
    # Bottom-Left Plot
    ax3 = subplot(2,2,3)
    I = array(I)
    L3, = plot(T[0:max_ind],
               I[0:max_ind],'c',lw=2,
               label="Predicted Infectious Population")         
    L3_1, = plot(T[0:max_ind],
               0.5*I[0:max_ind],'c-',lw=2,
               label="50% Infectious Population")
    L3_2, = plot(T[0:max_ind],
               0.33*I[0:max_ind],'c--',lw=2,
               label="33% Infectious Population")
    case_scatter, = plot(case_times,cases,'bo',label="El Paso Positive Test Results")      
    ax3.set_ylim(top=1.1*max(L3.get_ydata()))
    ax3.set_ylim(bottom=0)
    title("Predicted vs. Reported Infectious in El Paso County")
    grid(); legend()
    
    # Bottom-Right Plot
    ax4 = subplot(2,2,4)
    L4, = plot(T,I,'c',lw=2,label="Predicted Infectious Population")
    case_scatter2, = plot(case_times,cases,'bo',
                          label="El Paso Positive Test Results")      
    ax4.set_ylim(top=1.1*max(L4.get_ydata()))
    ax4.set_ylim(bottom=0)
    title("Predicted Infectious over the Long Term")
    grid(); legend()
    
    
    delta = 0.01
    axcolor = 'lightgoldenrodyellow'
    # First column of sliders
    ax_init_rho =       axes([0.1, 0.200, 0.32, 0.015], facecolor=axcolor)
    s_init_rho =        Slider(ax_init_rho, 'Initial R\u2080', 
                             0.5, 6, valinit=init_rho, valstep=delta)
    ax_lock_rho =       axes([0.1, 0.175, 0.32, 0.015], facecolor=axcolor)
    s_lock_rho =        Slider(ax_lock_rho, 'Stay-at-Home R\u2080', 
                             0.5, 6, valinit=lock_rho, valstep=delta)
    ax_post_lock_rho =  axes([0.1, 0.150, 0.32, 0.015], facecolor=axcolor)
    s_post_lock_rho =   Slider(ax_post_lock_rho, 'R\u2080 After Lockdown', 
                             0.5, 6, valinit=post_lock_rho, valstep=delta)    
    ax_post_lock_date = axes([0.1, 0.125, 0.32, 0.015], facecolor=axcolor)
    s_post_lock_date =  Slider(ax_post_lock_date, 'Stay-at-Home End (Days after 1 Mar)', 
                             50, 200, valinit=post_lock_time, valstep=10*delta) 
    ax_init_inf =       axes([0.1, 0.100, 0.32, 0.015], facecolor=axcolor)
    s_init_inf =        Slider(ax_init_inf, 'Infected on %s'%start_date.date(), 
                             1, 100, valinit=init_inf, valstep=10*delta) 
    ax_init_exp =       axes([0.1, 0.075, 0.32, 0.015], facecolor=axcolor)
    s_init_exp =        Slider(ax_init_exp, 'Exposed on %s'%start_date.date(), 
                             1, 100, valinit=0, valstep=10*delta)     
    ax_init_rec =       axes([0.1, 0.050, 0.32, 0.015], facecolor=axcolor)
    s_init_rec =        Slider(ax_init_rec, 'Recovered on %s'%start_date.date(), 
                             1, 100, valinit=0, valstep=10*delta)             
    # Second column of sliders
    ax_tau =            axes([0.58, 0.200, 0.32, 0.015], facecolor=axcolor)
    s_tau =             Slider(ax_tau, 'Infectious Period (Days)', 
                             1, 14, valinit=tau, valstep=delta)    
    ax_mu =             axes([0.58, 0.175, 0.32, 0.015], facecolor=axcolor)
    s_mu =              Slider(ax_mu,  'Incubation Period', 
                            0, 7,  valinit=mu,  valstep=delta)       
    if model in ['SIR_nu', 'SEIR_nu']:
        ax_het =        axes([0.58, 0.150, 0.32, 0.015], facecolor=axcolor)
        s_het =         Slider(ax_het,  'Mixing Heterogeneity (nu)', 
                                    1, 2.5,  valinit=nu,  valstep=delta)  
    else:
        ax_het =        axes([0.58, 0.150, 0.32, 0.015], facecolor=axcolor)
        s_het =         Slider(ax_het,  'Mixing Homogeneity (k)', 
                                    0.01, 10,  valinit=k,  valstep=delta)  
    ax_mort_rate =      axes([0.58, 0.125, 0.32, 0.015], facecolor=axcolor)
    s_mort_rate =       Slider(ax_mort_rate,  'Mortality Rate (%)', 
                            0.1, 10,  valinit=100*mort_rate,  valstep=delta)  
    ax_symp_2_death =   axes([0.58, 0.100, 0.32, 0.015], facecolor=axcolor)
    s_symp_2_death =    Slider(ax_symp_2_death,  'Symptom Onset to Death (days)', 
                            3, 20,  valinit=symp_2_death,  valstep=5*delta)  
    ax_conf_case_delay= axes([0.58, 0.075, 0.32, 0.015], facecolor=axcolor)
    s_conf_case_delay = Slider(ax_conf_case_delay,  'Testing Delay (days)', 
                            0, 10,  valinit=conf_case_delay,  valstep=5*delta) 
    ax_t=               axes([0.58, 0.050, 0.32, 0.015], facecolor=axcolor)
    s_t =               Slider(ax_t,  'Timestep (Minutes)', 
                            15, 60*24,  valinit=(24*60)*1/48,  valstep=5)                             
                              

    def update(val):
        model = radio.value_selected
        init_rho =      s_init_rho.val
        lock_rho =      s_lock_rho.val
        post_lock_rho = s_post_lock_rho.val
        post_lock_date= s_post_lock_date.val
        post_lock_time= post_lock_date
        post_lock_date= start_date + timedelta(days=post_lock_time)
        init_inf =      s_init_inf.val
        init_exp =      s_init_exp.val
        init_rec =      s_init_rec.val
        tau =           s_tau.val
        mu =            s_mu.val
        if model in ['SIR_nu','SEIR_nu']:
            nu =        s_het.val
        elif model == 'NBD_SEIR':
            k =         s_het.val
        mort_rate =     s_mort_rate.val/100
        symp_2_death =  s_symp_2_death.val
        conf_case_delay=s_conf_case_delay.val
        t =             (s_t.val)/(60*24)
        max_ind = int((today-start_date).days/t)
        [death_times, case_times, deaths, cases] = dataHandler(start_date, 
                                                    conf_case_delay)
        rho_sched = {lock_date:lock_rho,
                     post_lock_date:post_lock_rho}
        if model == 'SIR_nu':
            I_ind = 2; D_ind = 4
            out2 = SIR_nu(start_date,P=720403,I=init_inf,R=0,
                   rho=init_rho,tau=tau,nu=1.7,loc=loc,MaxDays=MaxDays,suppress_output=1,
                   rho_sched=rho_sched, mort_rate=mort_rate, symp_2_death=symp_2_death,t=t)
        elif model == 'SEIR_nu':
            I_ind = 3; D_ind = 5
            out2 = SEIR_nu(start_date,P=720403,E=0,I=init_inf,R=0, mu = mu,
                       rho=init_rho,tau=tau,nu=1.7,loc=loc,MaxDays=MaxDays,suppress_output=1,
                       rho_sched=rho_sched, mort_rate=mort_rate, symp_2_death=symp_2_death,t=t)
        elif model == 'NBD_SEIR':
            I_ind = 3; D_ind = 6
            out2 = NBD_SEIR(start_date,720403,init_exp,init_inf,init_rec,
                        init_rho,tau,k,mu,loc,MaxDays,suppress_output=1,
                        rho_sched=rho_sched, 
                        mort_rate=mort_rate, symp_2_death=symp_2_death, t=t)
        L1.set_xdata(out2[0][0:max_ind])
        L1.set_ydata(out2[D_ind][0:max_ind])
        ax1.set_ylim(top=1.1*max(L1.get_ydata()))
        ax1.set_ylim(bottom=0)
        L2.set_xdata(out2[0])
        L2.set_ydata(out2[D_ind])
        ax2.set_ylim(top=1.1*max(L2.get_ydata()))
        ax2.set_ylim(bottom=0)
        L3.set_xdata(out2[0][0:max_ind])
        L3.set_ydata(out2[I_ind][0:max_ind])
        L3_1.set_xdata(out2[0][0:max_ind])
        L3_1.set_ydata(0.5*array(out2[I_ind][0:max_ind]))
        L3_2.set_xdata(out2[0][0:max_ind])
        L3_2.set_ydata(0.33*array(out2[I_ind][0:max_ind]))
        case_scatter.set_xdata(case_times)
        case_scatter.set_ydata(cases)
        ax3.set_ylim(top=1.1*max(L3.get_ydata()))
        ax3.set_ylim(bottom=0)
        L4.set_xdata(out2[0])
        L4.set_ydata(out2[I_ind])
        case_scatter2.set_xdata(case_times)
        case_scatter2.set_ydata(cases)
        ax4.set_ylim(top=1.1*max(L4.get_ydata()))
        ax4.set_ylim(bottom=0)
        fig.canvas.draw_idle()
        
    s_init_rho.on_changed(update)    
    s_lock_rho.on_changed(update)
    s_post_lock_rho.on_changed(update)
    s_post_lock_date.on_changed(update)
    s_init_inf.on_changed(update) 
    s_init_exp.on_changed(update) 
    s_init_rec.on_changed(update) 
    s_tau.on_changed(update)
    s_mu.on_changed(update)
    s_k.on_changed(update)
    s_mort_rate.on_changed(update)
    s_symp_2_death.on_changed(update)
    s_conf_case_delay.on_changed(update)
    s_t.on_changed(update)
    
    resetax = axes([0.8, 0.001, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    def reset(event):
        s_init_rho.reset()    
        s_lock_rho.reset()
        s_post_lock_rho.reset()
        s_post_lock_date.reset()
        s_init_inf.reset() 
        s_init_exp.reset() 
        s_init_rec.reset() 
        s_tau.reset()
        s_mu.reset()
        s_k.reset()
        s_mort_rate.reset()
        s_symp_2_death.reset()
        s_conf_case_delay.reset()
        s_t.reset()
        return button
    button.on_clicked(reset)
    
    rax = axes([0.005, 0.75, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('SIR_nu', 
                               'SEIR_nu', 
                               'NBD_SEIR'), active=models[model])
    
    def model_switch(radio_model):
        update(1)
        fig.canvas.draw_idle()
        return radio
        
    radio.on_clicked(model_switch)
    
    show()
    #fig.savefig('demo.png', bbox_inches='tight')
    
#%matplotlib qt
Slider_NBD_SEIR()
