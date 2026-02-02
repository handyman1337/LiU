import numpy as np
from matplotlib import pyplot as plt
from utils import getpolicy, getvalue

from world import World

class GridWorld(World):

    def __init__(self, world_number):
        # Set world parameters
        self.worldNum = world_number
        self._xSize = 15
        self._ySize = 10
        if world_number == 1:
            self.__class__ = GridWorld1
        elif world_number == 2:
            self.__class__ = GridWorld2
        elif world_number == 3:
            self.__class__ = GridWorld3
        elif world_number == 4:
            self.__class__ = GridWorld4
        elif world_number == 5:
            self.__class__ = GridWorld5
        elif world_number == 6:
            self.__class__ = GridWorld6
        elif world_number == 7:
            self.__class__ = GridWorld7
        elif world_number == 8:
            self.__class__ = GridWorld8
        self.__init__()
        
    def getWorldSize(self):
        return (self._ySize, self._xSize)
    
    def getDimensionNames(self):
        return ["Y", "X"]
    
    def getActions(self):
        return ["Down", "Up", "Right", "Left"]
    
    def getState(self):
        return self._pos, (self._pos == self._term)
    
    def init(self):
        self._pos = tuple([np.random.choice(i-1) for i in self.getWorldSize()])
        while self._pos == self._term:
            self._pos = tuple([np.random.choice(i) for i in self.getWorldSize()])

    def doAction(self, act):
        if act not in self.getActions():
            print("Unknown action attempted")
            return False, []
        
        pos = list(self._pos)
        pos[0] += int(act=="Down")  - int(act=="Up")
        pos[1] += int(act=="Right") - int(act=="Left")

        if pos[0] >= self._ySize or pos[0] < 0 or pos[1] >= self._xSize or pos[1] < 0:
            valid = False
        else:
            valid = True
            self._pos = tuple(pos)
            
        reward = self._rewardMap[self._pos]
            
        return valid, reward
    
    def draw(self, epoch=None, Q=None, sleepTime=0, params={}):
        if Q is not None:
            P = getpolicy(Q)
            V = getvalue(Q)
        else:
            P = None
            V = None
            
        if V is None:
            plt.rcParams['figure.figsize']=(6.5,7)
        else:
            plt.rcParams['figure.figsize']=(14,7)
            
        self._drawPre()
        
        if V is not None:
            plt.subplot(1,2,1)
            
        plt.imshow(self._rewardMap, vmin=self._rclim[0], vmax=self._rclim[1])
        plt.plot(self._pos[1] , self._pos[0] , color='black', linewidth=2, marker='s', markerfacecolor='gray' , markersize=20)
        plt.plot(self._term[1], self._term[0], color='black', linewidth=2, marker='o', markerfacecolor='green', markersize=20)
        plt.colorbar(orientation="horizontal", pad=0.06)
        if P is None:
            plt.title('Reward map')
        else:
            plt.title('Reward map and policy')
                    
        if epoch is None:
            plt.suptitle(f'World {self.worldNum} "{self.Name}"', y=0.83)
        else:
            plt.suptitle(f'World {self.worldNum} "{self.Name}", Epoch {epoch}', y=0.83)
            
        if P is not None:
            self._plotarrows(P)
            
        if V is not None:
            plt.subplot(1,2,2)
            plt.imshow(V)
            plt.colorbar(orientation="horizontal", pad=0.06)
            plt.title('Value map')
        
        self._drawPost(sleepTime)
        
    def _plotarrows(self, P):
        """ PLOTARROWS
        Displays a policy matrix as an arrow in each state.
        """
        x,y = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))

        u = np.zeros(x.shape)
        v = np.zeros(y.shape)

        v[P==2] = 1
        v[P==3] = -1
        u[P==0] = -1
        u[P==1] = 1
        
        # Do not draw arrow for the terminal state
        u[self._term[0], self._term[1]] = 0
        v[self._term[0], self._term[1]] = 0

        plt.quiver(v,u,color='r')
        
        
class GridWorld1(GridWorld):
    def __init__(self):
        self.Name = "Annoying block"
        
    def init(self):
        self._rewardMap = -0.1 * np.ones((self._ySize, self._xSize))
        self._rewardMap[:8, 4:8] = -2.28
        self._term = (3, 12)
        self._rclim = (-2.3,-0.1)
        super().init()
        
        
class GridWorld2(GridWorld):
    def __init__(self):
        self.Name = "Annoying random block"
        
    def init(self):
        self._rewardMap = -0.1 * np.ones((self._ySize, self._xSize))
        if np.random.rand() < 0.2:
            self._rewardMap[:8, 4:8] = -11
        self._term = (3, 14)
        self._rclim = (-11,-0.1)
        super().init()
        
        
class GridWorld3(GridWorld):
    def __init__(self):
        self.Name = "Road to the pub"
        
    def init(self):
        self._rewardMap = -0.5 * np.ones((self._ySize, self._xSize))
        self._rewardMap[:3, :] = -0.01
        self._rewardMap[7, :] = -0.01
        self._rewardMap[:, :3] = -0.01
        self._rewardMap[:, 12:15] = -0.01
        self._term = (9,13)
        self._rclim = (-0.5,-0.01)
        self._pos = (7, 1)
        
        
class GridWorld4(GridWorld):
    def __init__(self):
        self.Name = "Road home from the pub"
        
    def init(self):
        self._rewardMap = -0.5 * np.ones((self._ySize, self._xSize))
        self._rewardMap[:3, :] = -0.01
        self._rewardMap[7, :] = -0.01
        self._rewardMap[:, :3] = -0.01
        self._rewardMap[:, 12:15] = -0.01
        self._term = (7,1)
        self._rclim = (-0.5,-0.01)
        self._pos = (9, 13)
        
    def doAction(self, act):
        if np.random.rand() < 0.3:
            act = np.random.choice(self.getActions())
        return super().doAction(act)
    
    
class GridWorld5(GridWorld):
    def __init__(self):
        self.Name = "Warpspace"
        
    def init(self):
        self._rewardMap = -np.ones((self._ySize, self._xSize))
        self._term = (6,13)
        self._rclim = (-2, 0)
        super().init()
        
    def doAction(self, act):
        v,r = super().doAction(act)
        if self._pos == (1,1):
            self._pos = (8,13)
        return v,r
    
class GridWorld6(GridWorld):
    def __init__(self):
        self.Name = "Torus"
        
    def init(self):
        self._rewardMap = -np.ones((self._ySize, self._xSize))
        self._term = (8,13)
        self._rclim = (-2, 0)
        super().init()
        
    def doAction(self, act):
        if act not in self.getActions():
            print("Unknown action attempted")
            return False, []
        pos = list(self._pos)
        pos[0] += int(act=="Down")  - int(act=="Up")
        pos[1] += int(act=="Right") - int(act=="Left")
        if pos[0] >= self._ySize:
            pos[0] = 0
        if pos[0] < 0:
            pos[0] = self._ySize-1
        if pos[1] >= self._xSize:
            pos[1] = 0
        if pos[1] < 0:
            pos[1] = self._xSize-1
        self._pos = tuple(pos)
        reward = self._rewardMap[self._pos]
        return True, reward
        
class GridWorld7(GridWorld):
    def __init__(self):
        self.Name = "Steps"
        
    def init(self):
        self._rewardMap = -0.01 * np.ones((self._ySize, self._xSize))
        for i in range(self._ySize):
            self._rewardMap[i,:-1] = -0.01 - (self._ySize - i) / 1000;
        self._term = (0,14)
        self._rclim = (-0.02,-0.01)
        super().init()

        
class GridWorld8(GridWorld):
    def __init__(self):
        self.Name = "Two layers"
        self._zSize = 2

    def getWorldSize(self):
        return (self._ySize, self._xSize, self._zSize)
        
    def getDimensionNames(self):
        return ["Y", "X", "Z"]
        
    def getActions(self):
        return ["Down", "Up", "Right", "Left", "ZUp", "ZDown"]
        
    def init(self):
        self._rewardMap = -0.01 * np.ones((self._ySize, self._xSize, self._zSize))
        #self._rewardMap[:,6:9,0] = -0.02
        #self._rewardMap[:,:4,1] = -0.02
        #self._rewardMap[:,11:,1] = -0.02
        self._rewardMap[:7 ,2:7  ,0] = -0.02
        self._rewardMap[3:7,3:13 ,0] = -0.02
        self._rewardMap[3: ,8:13,0] = -0.02
        self._rewardMap[:,:,1] = -0.02
        self._rewardMap[:,6:9,1] = -0.01
        self._term = (9,14,0)
        self._rclim = (-0.02,-0.01)
        super().init()
        
    def doAction(self, act):
        if act not in self.getActions():
            print("Unknown action attempted")
            return False, []
        
        pos = list(self._pos)
        pos[0] += int(act=="Down")  - int(act=="Up")
        pos[1] += int(act=="Right") - int(act=="Left")
        pos[2] += int(act=="ZUp")   - int(act=="ZDown")
                
        if pos[0] >= self._ySize or pos[0] < 0 or pos[1] >= self._xSize or pos[1] < 0 or pos[2] >= self._zSize or pos[2] < 0:
            valid = False
        else:
            valid = True
            self._pos = tuple(pos)
            
        reward = self._rewardMap[self._pos]
            
        return valid, reward
    
    def draw(self, epoch=None, Q=None, sleepTime=0.01, params={}):
        if Q is not None:
            P = getpolicy(Q)
            V = getvalue(Q)
        else:
            P = None
            V = None
            
        if V is None:
            plt.rcParams['figure.figsize']=(6.5,7)
        else:
            plt.rcParams['figure.figsize']=(14,7)
            
        self._drawPre()
        
        if V is not None:
            plt.subplot(1,2,1)
            
        z = self._pos[-1]
        plt.imshow(self._rewardMap[:,:,z], vmin=self._rclim[0], vmax=self._rclim[1])
        plt.plot(self._pos[1] , self._pos[0] , color='black', linewidth=2, marker='s', markerfacecolor='gray' , markersize=20)
        if (z == 0):
            plt.plot(self._term[1], self._term[0], color='black', linewidth=2, marker='o', markerfacecolor='green', markersize=20)
        plt.colorbar(orientation="horizontal", pad=0.06)
        if P is None:
            plt.title('Reward map')
        else:
            plt.title('Reward map and policy')
                    
        if epoch is None:
            plt.suptitle(f'World {self.worldNum} "{self.Name}"', y=0.83)
        else:
            plt.suptitle(f'World {self.worldNum} "{self.Name}", Epoch {epoch}', y=0.83)
            
        if P is not None:
            self._plotarrows(P[:,:,z], z)
            
        if V is not None:
            plt.subplot(1,2,2)
            plt.imshow(V[:,:,z])
            plt.colorbar(orientation="horizontal", pad=0.06)
            plt.title('Value map')
        
        self._drawPost(sleepTime)
        
    def _plotarrows(self, P, z):
        """ PLOTARROWS
        Displays a policy matrix as an arrow in each state.
        """
        x,y = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))

        u = np.zeros(x.shape)
        v = np.zeros(y.shape)

        v[P==2] = 1
        v[P==3] = -1
        v[P==4] = 0.7
        v[P==5] = 0.7
        
        u[P==0] = -1
        u[P==1] = 1
        u[P==4] = 0.7
        u[P==5] = -0.7

        # Do not draw arrow for the terminal state
        if z == self._term[2]:
            u[self._term[0], self._term[1]] = 0
            v[self._term[0], self._term[1]] = 0
        
        plt.quiver(v,u,color='r')
        
        