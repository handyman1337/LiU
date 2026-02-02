
import time
import IPython.display as display
import pylab as pl

from abc import ABC, abstractmethod

class World(ABC):
    '''
    World, abstract base class for all other worlds in this RL framework.
    Declares the following methods that must be implemented by child classes:
        - getWorldSize: Returns the size of the state space as a tuple
        - getDimensionNames: Returns a list of names the state dimensions
        - getActions: Returns a list of action names
        - init: Initialize the state for a new epoch
        - getState: Retruns the current state of the World, and if this state is terminal
        - doAction: Performes the specified action and updates the state
        - draw: Updates any visual information
    '''
    
    def __init__(self):
        pass
    
    @abstractmethod
    def getWorldSize(self):
        pass
    
    @abstractmethod
    def getDimensionNames(self):
        pass
    
    @abstractmethod
    def getActions(self):
        pass
    
    @abstractmethod
    def init(self):
        pass
    
    @abstractmethod
    def getState(self):
        pass
        
    @abstractmethod
    def doAction(self, act):
        pass
    
    @abstractmethod
    def draw(self, epoch=None, Q=None, sleepTime=0.01, params={}):
        pass
        
    def _drawPre(self):
        pl.clf()
        
    def _drawPost(self, sleepTime=0):
        display.display(pl.gcf())
        display.clear_output(wait=True)
        time.sleep(sleepTime)