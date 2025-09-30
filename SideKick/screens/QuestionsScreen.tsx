import React, { useState } from "react";
import { View, Text, StyleSheet, TouchableOpacity, TextInput } from "react-native"; 
import { SafeAreaView } from "react-native-safe-area-context";
const BELT_COLORS = ['Beginner', 'White', 'Yellow', 'Green', 'Blue', 'Red', 'Black'];

export default function QuestionsScreen({ navigation }: { navigation: any }) {
    const [name, setName] = useState(""); // hold user name
    const [selectedBelt, setSelectedBelt] = useState(""); // hold selected belt color

    return (
        <SafeAreaView style= {styles.safeArea}>
            <View style= {styles.container}>
                <Text style= {styles.header}>Ready to kick off?</Text>
                
                <Text style= {styles.label}>What's your name?</Text>
                <TextInput
                    style= {styles.input}
                    placeholder= "Enter your name"
                    placeholderTextColor= "#777"
                    value= {name}
                    onChangeText= {setName}
                />

                <Text style= {styles.label}>Which Taekwondo belt do you have right now?</Text>
                {BELT_COLORS.map((color) => (
                    <TouchableOpacity
                        key= {color}
                        style= {[styles.beltButton, selectedBelt == color && styles.selectedBeltButton]}
                        onPress= {() => setSelectedBelt(color)}
                    >
                        <Text style= {[styles.beltButtonText, selectedBelt == color && styles.selectedBeltButtonText]}>
                            {color}
                        </Text>
                    </TouchableOpacity>
                ))}
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safeArea: { flex: 1, backgroundColor: '#121212' },
    container: { padding: 20 },
    header: { color: 'white', fontSize: 24, fontWeight: 'bold', marginBottom: 30 },
    label: { fontSize: 16, color: '#AAA', marginBottom: 10, marginTop: 20 },
    input: {
        backgroundColor: '#333',
        color: 'white',
        padding: 15,
        borderRadius: 8,
        fontSize: 16,
    },
    beltButton: {
        backgroundColor: '#333',
        padding: 15, 
        borderRadius: 8, 
        marginBottom: 10, 
    }, 
    selectedBeltButton: {
        backgroundColor: 'white', 
    },
    beltButtonText: {
        color: 'white',
        fontSize: 16,
    }, 
    selectedBeltButtonText: {
        color: 'black',
    },
});