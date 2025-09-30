import React, { useState } from "react";
import { View, Text, TextInput, StyleSheet, ImageBackground, TouchableOpacity, StatusBar } from "react-native";

const backgroundImage = require("../assets/onboarding-bg.png");

export default function OnboardingScreen({ navigation }: { navigation: any }) {
    const [name, setName] = useState("");

    return (
        <ImageBackground source={backgroundImage} style={styles.background}>
            <StatusBar barStyle="light-content" />
            <View style={styles.container}>
                <Text style={styles.title}>Welcome to SideKick</Text>
                <TouchableOpacity
                    style={styles.button}
                    onPress = {() => navigation.navigate('Questions')}
                >
                    <Text style= {styles.buttonText}>GET STARTED</Text>
                </TouchableOpacity>
            </View>
        </ImageBackground>
    );
}

const styles = StyleSheet.create({
    background: {
        flex: 1, // fits the whole screen
        justifyContent: 'flex-end', // positions text at bottom
    }, 
    container: {
        padding: 30, 
        paddingBottom: 60, 
    }, 
    title: {
        color: 'white',
        fontSize: 48, 
        fontWeight: 'bold', 
        marginBottom: 20,
    }, 
    button: {
        backgroundColor: 'white', 
        paddingVertical: 15,
        borderRadius: 8, 
        alignItems: 'center',
    }, 
    buttonText: {
        color: 'black', 
        fontSize: 18,
        fontWeight: 'bold',
    },
}); 