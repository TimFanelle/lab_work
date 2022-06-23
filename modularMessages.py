from ctypes import *
import struct
import enum

#define enum here
class messageTypes(enum.Enum):
    sync = 1
    trigger = 2
    heartbeat = 3
    ping = 4
    csv = 5
    onlyFloat = 6
    alphanumeric = 7

def sendMessage(msgType, headers, *inputs):
    #this needs to be modified to account for message type
    message = bytearray()
    headers = headers.split('%')[1:]
    msgType = c_short(msgType.value)
    headerBytes = 0
    msgBytes = 0
    header = bytearray()
    msgData = bytearray()
    for i in range(len(headers)):
        currHead = headers[i]
        currInput = inputs[i]
        if isinstance(currInput, (float, int, str)):
            header.extend(headers[i].encode('utf-8'))
            if currHead == 'f' or currHead == 'd':
                headerBytes += 1
                msgBytes += 4
                if currHead == 'd' and isinstance(currInput, int):
                    msgData.extend(c_int(currInput))
                elif isinstance(currInput, float):
                    msgData.extend(c_float(currInput))
                else:
                    raise ValueError("Incorrect input at the %sth place for given header", i)
            elif currHead == 's' and isinstance(currInput, str):
                headerBytes += 3
                strLen = len(currInput)
                header.extend(c_short(strLen))
                for letter in currInput:
                    msgData.extend(letter.encode('utf-8'))
            else:
                raise ValueError("Incorrect input at the %sth place for given header", i)
    headerBytes = c_short(headerBytes)
    msgBytes = c_uint(msgBytes)
    
    #finalize message definition
    message.extend(msgType)
    message.extend(headerBytes)
    message.extend(msgBytes)
    message.extend(header)
    message.extend(msgData)
    print(message)
    return message

def receiveMsg(msg):
    msgType = int.from_bytes(msg[:2], "little")
    headerSize = int.from_bytes(msg[2:4], "little")
    msgSize = int.from_bytes(msg[4:8], "little")

    received_items = []

    #this needs to be modified to account for message type
    #decode the header
    header = msg[8:8+headerSize].decode('utf-8')
    currStart = 8+headerSize

    u = 0
    while u < len(header):
        curHead = header[u]
        if curHead == 'f':
            received_items.append(struct.unpack('<f',msg[currStart:currStart+4])[0])
            currStart += 4
            u += 1
        elif curHead == 'd':
            received_items.append(int.from_bytes(msg[currStart:currStart+4], "little"))
            currStart += 4
            u += 1
        else:
            l = int.from_bytes(header[u+1:u+3].encode('utf-8'), "little")
            received_items.append(msg[currStart:currStart+l].decode('utf-8'))
            u += 3
            currStart += l
    print(received_items)

receiveMsg(sendMessage(messageTypes.alphanumeric, "%f%f%f%d%s%f", 13.2, 14.2, 15.3, 12, "Hellow", 16.2))