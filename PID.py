#-*- coding: UTF-8 -*-
# 调用必需库
import time


class PID:
    def __init__(self, kP=1, kI=0, kD=0, output_bound_low=-5, output_bound_high=5):
        # 初始化PID参数
        self.kP = kP
        self.kI = kI
        self.kD = kD
        
        # 输出钳位 单位是角度 
        self.output_bound_low = output_bound_low
        self.output_bound_high = output_bound_high
        
        self.error_threhold = 5
        
        # PID输出是否饱和的flag
        self.output_windup_flag = False
        
        # PID输出是否与error同号 同号会导致输出饱和更加严重
        self.error_output_sign_flag = False

        # 初始化当前时间和上一次计算的时间
        self.currTime = time.time()
        self.prevTime = self.currTime

        # 初始化上一次计算的误差
        self.prevError = 0
        
        # 初始化上一次的输出值
        self.prevOutput = 0

        # 初始化误差的比例值，积分值和微分值
        self.cP = 0
        self.cI = 0
        self.cD = 0


    def update(self, error, sleep=0.1):
        # pid更新间隔
        time.sleep(sleep)

        # 获取当前时间并计算时间差
        self.currTime = time.time()
        deltaTime = self.currTime - self.prevTime

        # 计算误差的微分
        deltaError = error - self.prevError

        # P 比例项
        self.cP = error

        # D 微分项
        self.cD = (deltaError / deltaTime) if deltaTime > 0 else 0
        
        # 判断PID输出是否超限
        self.output_windup_flag = self.prevOutput > self.output_bound_high or self.prevOutput < self.output_bound_low
            
        # 判断error和PID输出是不是同号
        self.error_output_sign_flag = (self.prevOutput >= 0 and error >= 0) or (self.prevOutput < 0 and error < 0)
        
        # I 积分项 采用条件积分
        # 输出已经饱和 且 Error和输出同号会导致饱和更加严重 因此切断积分器
        if self.output_windup_flag and self.error_output_sign_flag:
            self.cI += 0
        # 开启积分器
        else:
            # 积分项
            self.cI += error * deltaTime
            
#             if error <= self.error_threhold:
#                 self.cI += 0
        
        # print('cI: ' + str(self.cI))
        print('error' + str(error))

        # 保存时间和误差为下次更新做准备
        self.prevTime = self.currTime
        self.prevError = error
        
        # 变速积分 error大积分小 error小积分大
        scale_kI = self.kI / (1 + 0.0001 * abs(error))
        
        # 计算输出
        Output = sum([self.kP * self.cP, scale_kI * self.cI, self.kD * self.cD])
        # 保存输出为下一次更新准备
        self.prevOutput = Output

        # 返回输出值
        return Output
    